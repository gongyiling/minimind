"""
预训练数据预 tokenize 脚本

将 jsonl 文本预先编码为 token id 矩阵，保存为 uint16 的 .npy 文件，
训练时由 PretokenizedDataset 直接读取，省去每个 epoch 重复分词的开销。

用法（从 trainer 目录执行）：
    python ../dataset/pretokenize.py --max_seq_len 340
    python ../dataset/pretokenize.py --input ../dataset/pretrain_t2t_mini.jsonl --output ../dataset/pretrain_t2t_mini_tokenized.npy
"""
import os
import sys
import json
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize pretrain jsonl data")
    parser.add_argument("--input", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="输入 jsonl 路径")
    parser.add_argument("--output", type=str, default="../dataset/pretrain_t2t_mini_tokenized.npy", help="输出 npy 路径")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer 目录")
    parser.add_argument("--max_seq_len", type=int, default=340, help="每条样本固定长度（与训练时一致）")
    parser.add_argument("--batch_size", type=int, default=8192, help="批量编码的样本数")
    parser.add_argument("--log_interval", type=int, default=50, help="每多少个 batch 打印一次进度")
    return parser.parse_args()


def count_lines(path):
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    if args.max_seq_len < 3:
        raise ValueError("max_seq_len 至少为 3（bos + 1 token + eos）")

    print(f"[1/3] 统计样本数: {args.input}")
    total = count_lines(args.input)
    print(f"      共 {total:,} 条样本, max_seq_len={args.max_seq_len}, dtype=uint16")

    print(f"[2/3] 加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("tokenizer 缺少 bos/eos/pad token")
    if max(bos_id, eos_id, pad_id) > np.iinfo(np.uint16).max:
        raise ValueError("token id 超出 uint16 范围，请改用 uint32")
    print(f"      bos={bos_id}, eos={eos_id}, pad={pad_id}, vocab_size={len(tokenizer)}")

    print(f"[3/3] 开始 tokenize -> {args.output}")
    # 用 memmap 流式写入，避免把整个矩阵放进内存
    out = np.lib.format.open_memmap(
        args.output, mode='w+', dtype=np.uint16, shape=(total, args.max_seq_len)
    )

    start_time = time.time()
    written = 0
    batch_texts = []
    batch = []

    with open(args.input, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)['text']
            except (KeyError, json.JSONDecodeError):
                # 脏数据：写成全 pad 行，保持样本数对齐
                text = ""
            batch_texts.append(str(text))
            batch.append(line_no)

            if len(batch_texts) >= args.batch_size:
                _encode_batch(batch_texts, out, written, tokenizer, args.max_seq_len, bos_id, eos_id, pad_id)
                written += len(batch_texts)
                batch_texts, batch = [], []
                if (written // args.batch_size) % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    speed = written / elapsed
                    eta = (total - written) / max(speed, 1e-6) / 60
                    print(f"      {written:,}/{total:,} ({written / total * 100:.1f}%), "
                          f"{speed:,.0f} 条/s, 预计剩余 {eta:.1f}min", flush=True)

        if batch_texts:
            _encode_batch(batch_texts, out, written, tokenizer, args.max_seq_len, bos_id, eos_id, pad_id)
            written += len(batch_texts)

    out.flush()
    del out

    # 元数据：供 PretokenizedDataset 校验一致性
    meta = {
        "num_samples": written,
        "max_seq_len": args.max_seq_len,
        "dtype": "uint16",
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "source": os.path.basename(args.input),
    }
    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"[完成] 写入 {written:,} 条, 耗时 {elapsed / 60:.1f}min, 文件大小 {size_mb:.0f}MB")
    print(f"       数据: {args.output}")
    print(f"       元数据: {meta_path}")
    print(f"       训练时加参数: --pretokenized_path {args.output}")


def _encode_batch(texts, out, offset, tokenizer, max_seq_len, bos_id, eos_id, pad_id):
    """把一批文本编码后写入 memmap 的指定偏移。"""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len - 2,  # 预留 bos/eos 位置，与原 Dataset 行为一致
    )['input_ids']

    for i, ids in enumerate(encoded):
        seq = [bos_id] + ids + [eos_id]
        row = out[offset + i]
        row[:] = pad_id
        n = min(len(seq), max_seq_len)
        row[:n] = seq[:n]


if __name__ == "__main__":
    main()
