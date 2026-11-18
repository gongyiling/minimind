import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset, PretokenizedDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    iter_end_time = start_time  # 上一步训练结束时间，用于统计数据加载耗时
    data_time_sum = 0.0  # 距上次日志的累计数据加载等待时间
    last_log_time, last_log_step = start_time, start_step  # 上次日志的时间与 step（滑动窗口）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        data_time = time.time() - iter_end_time  # 等待并加载本批次数据的时间
        data_time_sum += data_time
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step == start_step + 1 or step % args.log_interval == 0 or step == iters:
            now = time.time()
            steps_delta = max(step - last_log_step, 1)
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            step_time = (now - last_log_time) / steps_delta  # 最近窗口的平均单步耗时
            avg_data_time = data_time_sum / steps_delta  # 最近窗口的平均数据加载等待
            eta_min = step_time * (iters - step) // 60
            log_msg = (f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                       f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                       f'lr: {current_lr:.8f}, step_time: {step_time:.3f}s, data_time: {avg_data_time:.3f}s, '
                       f'epoch_time: {eta_min:.1f}min')
            if args.log_memory and 'cuda' in args.device:
                log_msg += (f', mem: {torch.cuda.memory_allocated(args.device) / 1024 ** 3:.2f}GB/'
                            f'{torch.cuda.max_memory_allocated(args.device) / 1024 ** 3:.2f}GB')
            Logger(log_msg)
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            last_log_time, last_log_step, data_time_sum = now, step, 0.0

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        iter_end_time = time.time()
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（step）")
    parser.add_argument("--log_memory", type=int, default=1, choices=[0, 1], help="日志中是否打印GPU显存占用（0=否，1=是）")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--pretokenized_path', type=str, default='../dataset/pretrain_t2t_mini_tokenized.npy', help="预tokenize的npy路径（设置后跳过实时分词，优先于--data_path）")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 预 tokenize 数据是 mmap 读取，无需 worker；Windows 下多 worker 会因 spawn 复制内存导致占用暴涨
    if args.pretokenized_path and args.num_workers != 0:
        Logger(f'[配置] 检测到预tokenize数据，num_workers 从 {args.num_workers} 强制设为 0（避免内存暴涨）')
        args.num_workers = 0

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() and 'cuda' in args.device else 'N/A'
    Logger(f'[1/8] 环境与随机种子就绪: device={args.device} ({gpu_name}), 分布式={dist.is_initialized()}, rank={dist.get_rank() if dist.is_initialized() else 0}')
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    Logger('[2/8] 配置模型参数与检查断点...')
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    Logger(f'[2/8] 参数配置完成: hidden_size={args.hidden_size}, layers={args.num_hidden_layers}, use_moe={bool(args.use_moe)}' + (f', 检测到断点(epoch={ckp_data["epoch"]}, step={ckp_data.get("step", 0)})' if ckp_data else ', 无断点'))
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    Logger(f'[3/8] 设备与精度: device={args.device} (type={device_type}), dtype={args.dtype}, autocast={"off" if device_type == "cpu" else "on"}')
    if device_type == "cpu":
        Logger('[3/8] 警告: 当前以 CPU 训练。若机器有 NVIDIA 显卡却出现此提示，说明安装的 PyTorch 不带 CUDA 支持（torch.cuda.is_available()=False），请安装 CUDA 版 torch')
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
        Logger(f'[4/8] SwanLab 已就绪: project={args.wandb_project}, run={wandb_run_name}')
    
    # ========== 5. 定义模型、数据、优化器 ==========
    Logger('[5/8] 正在初始化模型...')
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    data_source = args.pretokenized_path if args.pretokenized_path else args.data_path
    Logger(f'[5/8] 模型就绪, 正在加载数据集: {data_source}')
    data_start = time.time()
    if args.pretokenized_path:
        train_ds = PretokenizedDataset(args.pretokenized_path, max_length=args.max_seq_len)
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    Logger(f'[5/8] 数据集加载完成: {len(train_ds)} 条样本, 耗时 {time.time() - data_start:.1f}s')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'[6/8] 断点恢复完成: 从 epoch={start_epoch + 1}, step={start_step} 继续')
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('[7/8] torch.compile 已开启')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger(f'[7/8] 分布式训练已启用: world_size={dist.get_world_size()}')

    # 预热编译：用 dummy batch 提前触发 torch.compile，避免编译开销混入第一个真实 step
    if args.use_compile == 1:
        Logger('[7/8] 预热编译中（dummy batch 跑一次完整前反向），首次约几分钟...')
        compile_start = time.time()
        dummy_ids = torch.randint(0, lm_config.vocab_size, (args.batch_size, args.max_seq_len), device=args.device)
        model.train()
        with autocast_ctx:
            warmup_res = model(dummy_ids, labels=dummy_ids)
            warmup_loss = warmup_res.loss + warmup_res.aux_loss
        scaler.scale(warmup_loss).backward()
        optimizer.zero_grad(set_to_none=True)  # 只触发编译，不更新参数，避免污染训练状态
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        del dummy_ids, warmup_res, warmup_loss
        Logger(f'[7/8] 预热编译完成, 耗时 {time.time() - compile_start:.1f}s')

    # ========== 8. 开始训练 ==========
    Logger(f'[8/8] 训练配置: epochs={args.epochs}, batch_size={args.batch_size}, accumulation_steps={args.accumulation_steps}, '
           f'lr={args.learning_rate}, log_interval={args.log_interval}, save_interval={args.save_interval}, '
           f'device={args.device}, dtype={args.dtype}, data_path={args.data_path}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f'===== Epoch [{epoch + 1}/{args.epochs}] 开始, 数据集大小={len(train_ds)}, 本轮step={len(loader) + skip} =====')
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
        Logger(f'===== Epoch [{epoch + 1}/{args.epochs}] 结束 =====')

    Logger(f'训练完成，共 {args.epochs} 个epoch')
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()