#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#  blip2-flan-t5-xl     blip2-opt-2.7b

BLIP-2 Fine-tuning Script for Fake/Real Classification (Answer-only loss)
=========================================================================
- 任务：根据图片回答 'fake' 或 'real'
- 训练：输入统一 prompt + 答案文本（fake/real），但仅对“答案部分的 token”计算损失
- 特性：LoRA、FP16、显存 OOM 自适应、可选梯度检查点
# 修改  124行  改228  87  186

ADM  DDPM  Diff-ProjectedGAN   Diff-StyleGAN2  IDDPM   LDM  PNDM  ProGAN  ProjectedGAN  StyleGAN
                ./test_debug.csv \        ./data/Train_CSV/train_LDM.csv \
示例命令：
python blip2_detect-005.py \
  --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
  --epochs 5 \
  --lr 5e-5 \
  --batch_size 32 \
  --save_path ./SaveFineTune/LDM-train-epochs05 \
  --fp16
"""

import os
import argparse
# import args
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import peft

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
# from sklearn.utils import resample  #后加


# =========================
# 基础：随机种子
# =========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# collate：与现有 dataset.py 适配
# =========================
def collate_fn_factory(processor):
    def collate_fn(batch):
        out = {}
        for k in batch[0].keys():
            if k != "text":
                out[k] = torch.stack([b[k] for b in batch])
            else:
                out["text"] = [b["text"] for b in batch]
        return out
    return collate_fn

# =========================
# 显存自适应的 batch 估计
# =========================
def estimate_initial_batch_size(fp16=False, base_batch=4, safety_factor=0.8):
    if not torch.cuda.is_available():
        return base_batch
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory
    avail = int(total * safety_factor)
    est_per_sample = 3 * 1024 ** 3  # 经验值
    if fp16:
        est_per_sample //= 2
    bs = base_batch
    while bs * est_per_sample > avail and bs > 1:
        bs //= 2
    return max(1, bs)

# =========================
# 构造“仅答案监督”的 labels
# =========================
def build_answer_only_labels(tokenizer, prompt_texts, answer_texts, device):
    """
    将 prompt + answer 一起分词作为输入；但把 prompt 部分 label 置为 -100，仅对 answer 计算损失。
    """
    # tokenize full target (prompt + answer)
    full_texts = [p + a for p, a in zip(prompt_texts, answer_texts)]
    # tok_full = tokenizer(full_texts, padding=True, return_tensors="pt")
    tok_full = tokenizer(full_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = tok_full.input_ids.to(device)

    # 计算每条样本 prompt 的 token 长度
    tok_prompt = tokenizer(prompt_texts, padding=False, return_tensors=None)
    prompt_lens = []
    for ids in tok_prompt["input_ids"]:
        prompt_lens.append(len(ids))

    # 构造 labels：复制 input_ids，再把 prompt 段置为 -100
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100  # 忽略 prompt 部分的 loss

    return input_ids, labels, tok_full.attention_mask.to(device)

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 for Fake/Real (answer-only loss)")
    parser.add_argument("--dataset", type=str, required=True, help="Training CSV with columns: image,text (fake/real)")
    parser.add_argument("--model_path", type=str, default="./blip2-opt-2.7b", help="BLIP-2 base model path")
    parser.add_argument("--save_path", type=str, default="./SaveFineTune", help="Where to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)     #5e-5
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--enable_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj")
    # parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 1) 模型与处理器
    print("📦 Loading BLIP-2 and processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    tokenizer = processor.tokenizer  # 便于手动分词
    # model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, device_map="auto")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype=torch.float16  # 新写法，更稳
    )

    model.config.use_cache = False
#后加
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False

    # LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules
    )
    model = peft.get_peft_model(model, lora_cfg)

    # stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    if args.enable_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled.")
        except Exception:
            print("⚠️ Gradient checkpointing not supported for this model.")

    # 2) 数据
    from dataset import ImageCaptioningDataset

    print(f"📂 Reading CSV: {args.dataset}")
    df = pd.read_csv(args.dataset)
    # ✅ 清理无效标签行    #后加 改
    df = df.dropna(subset=['text'])  # 去除 text 为 NaN 的行
    df = df[df['text'].str.strip() != '']  # 去除 text 为空字符串的行
    df["text"] = df["text"].astype(str).str.lower().str.strip()  # 规范化标签（统一小写）
    print("✅ 清洗后标签分布：")
    print(df["text"].value_counts())

    if "image" not in df.columns or "text" not in df.columns:
        raise ValueError("❌ CSV 必须包含 'image' 和 'text' 两列；text 需为 'fake' 或 'real'")

    # 规范化标签
    # df["text"] = df["text"].astype(str).str.lower().str.strip()
    # print("✅ 标签分布：\n", df["text"].value_counts())

    train_dataset = ImageCaptioningDataset(df, processor)
    collate_fn = collate_fn_factory(processor)

    # 3) 训练准备
    bs = estimate_initial_batch_size(fp16=args.fp16, base_batch=args.batch_size)
    print(f"🧮 Using batch_size = {bs}")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    # scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)   #改
    model.train()

    # 统一的训练提示词（与测试脚本保持一致，含 no extra words 约束）
    base_prompt = "Is this image fake or real? Answer ONLY with 'fake' or 'real' (no extra words). "

    # 4) 训练循环
    for epoch in range(args.epochs):
        dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            try:
                images = batch["pixel_values"].to(device)
                labels_text = [("fake" if t.lower().strip().startswith("fake") else "real") for t in batch["text"]]

                # 构造每条样本的 prompt 与 (prompt+answer) 文本
                prompts = [base_prompt for _ in labels_text]
                answers = [("fake" if t == "fake" else "real") for t in labels_text]  # 防御性规范化
                # 注意：我们把 prompt 与 answer 合并给“语言侧输入”，仅监督 answer 部分
                # 手动构造语言侧 input_ids 与 labels（仅答案部分计算 loss）
                text_input_ids, text_labels, text_attn = build_answer_only_labels(
                    tokenizer=tokenizer,
                    # prompt_texts=prompts,
                    prompt_texts=[base_prompt] * len(labels_text),
                    answer_texts=answers,
                    device=device
                )

                # 视觉 + 语言 一起构造输入；此处我们把“语言侧 token”作为 text 来喂给模型
                # 重要：使用 processor 取视觉特征，语言侧 id 我们手工传入
                # vision_inputs = processor(images=images, return_tensors="pt")    # 改
                # vision_inputs = {k: v.to(device) for k, v in vision_inputs.items()}    # 改
                # 已在 ImageCaptioningDataset 中处理像素，无需重复处理
                vision_inputs = {"pixel_values": images}

                # 混合精度
                # with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=args.fp16):  # 改
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    outputs = model(
                        pixel_values=vision_inputs["pixel_values"],
                        input_ids=text_input_ids,
                        attention_mask=text_attn,
                        labels=text_labels,              # 仅答案部分有监督
                    )
                    loss = outputs.loss

                if torch.isnan(loss):          # 后加
                    print("⚠️ Loss is NaN, batch skipped")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                # if args.fp16:
                #     scaler.scale(loss).backward()
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                #     loss.backward()
                #     optimizer.step()
                if args.fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # 反缩放，防止爆梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if torch.isnan(loss):  # 后加
                        print("⚠️ Loss is NaN, skipping this batch")  # 后加
                        optimizer.zero_grad(set_to_none=True)    # 后加
                        continue    # 后加

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()


                pbar.set_postfix({"loss": float(loss.detach().cpu())})

            except RuntimeError as e:
                # 显存 OOM 自适应
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    old = bs
                    bs = max(1, bs // 2)
                    print(f"⚠️ CUDA OOM: batch_size {old} -> {bs}。将以更小 batch 继续")
                    break
                else:
                    raise

        print(f"✅ Epoch {epoch+1} finished.")

    # 5) 保存 LoRA 权重
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    print(f"🎯 模型已保存至: {args.save_path}")
