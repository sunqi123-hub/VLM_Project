#!/usr/bin/env python3
"""
BLIP-2 Fine-tuning Script for Fake/Real Classification
基于 BLIP-2 + LoRA 的合成图像检测任务微调脚本
"""
"""
ADM  DDPM  Diff-ProjectedGAN   Diff-StyleGAN2  IDDPM   LDM  PNDM  ProGAN  ProjectedGAN  StyleGAN
训练命令：

python blip2_detect-002.py \
  --dataset ./data/Train_CSV/train_LDM.csv \
  --epochs 2 \
  --batch_size 4 \
  --save_path ./SaveFineTune/LDM-train-epochs02 \
  --fp16

"""
# 修改blip2_detect-001，测试脚本就能理解模型输出的“fake/real”含义，准确率会大幅提升，F1 与 AUC 不再恒为 0 或 0.5

import os
import math
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import peft

# ===============================
# 基础设置
# ===============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn_factory(processor):
    def collate_fn(batch):
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                processed_batch["text"] = [example["text"] for example in batch]
        return processed_batch
    return collate_fn


def estimate_initial_batch_size(fp16=False, base_batch=4, safety_factor=0.8):
    if not torch.cuda.is_available():
        return base_batch
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory
    avail = int(total * safety_factor)
    est_per_sample = 3 * 1024 ** 3
    if fp16:
        est_per_sample //= 2
    batch = base_batch
    while batch * est_per_sample > avail and batch > 1:
        batch //= 2
    return max(1, batch)

# ===============================
# 主训练逻辑
# ===============================
if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 for Fake/Real Image Classification")
    parser.add_argument('--dataset', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--model_path', type=str, default='./blip2-opt-2.7b', help='Base BLIP2 model path')
    parser.add_argument('--save_path', type=str, default='./SaveFineTune', help='Path to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--enable_checkpointing', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--target_modules', type=str, default='q_proj,k_proj')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # ===============================
    # 加载模型和处理器
    # ===============================
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, device_map='auto')
    model.config.use_cache = False

    # LoRA 配置
    target_modules = [m.strip() for m in args.target_modules.split(',')]
    lora_config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        target_modules=target_modules
    )
    model = peft.get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.4f}%)")

    # 启用梯度检查点（可选）
    if args.enable_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        except Exception:
            print("Gradient checkpointing not supported for this model.")

    # ===============================
    # 数据加载
    # ===============================
    from dataset import ImageCaptioningDataset
    data = pd.read_csv(args.dataset)
    train_dataset = ImageCaptioningDataset(data, processor)
    collate_fn = collate_fn_factory(processor)

    init_batch = estimate_initial_batch_size(fp16=args.fp16, base_batch=args.batch_size)
    batch_size = init_batch
    print(f"Using batch size: {batch_size}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.train()
    prompt_text = "Is this image fake or real? Answer ONLY with 'fake' or 'real'."

    # ===============================
    # 训练循环
    # ===============================
    for epoch in range(args.epochs):
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            try:
                images = batch["pixel_values"].to(device)
                labels = batch["text"]  # 'fake' 或 'real'

                # 构造输入 prompt
                inputs = processor(
                    images=images,
                    text=[prompt_text] * len(labels),
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # 目标标签 (text → token)
                label_tokens = processor.tokenizer(
                    labels,
                    padding=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', enabled=args.fp16):
                    outputs = model(**inputs, labels=label_tokens)
                    loss = outputs.loss

                if args.fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                pbar.set_postfix({"loss": float(loss.detach().cpu())})

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    old_bs = batch_size
                    batch_size = max(1, batch_size // 2)
                    print(f"⚠️ OOM: batch size {old_bs} → {batch_size}")
                    break
                else:
                    raise

    # ===============================
    # 保存模型
    # ===============================
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    print(f"✅ 模型已保存至: {args.save_path}")
