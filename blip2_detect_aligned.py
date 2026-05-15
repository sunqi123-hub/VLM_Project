"""
按 blip2_test006.py 的测试逻辑来设计的
./data/Train_CSV_Balanced/train_LDM_balanced.csv \    ./data/Train_CSV/train_LDM.csv
 LDM   ADM  DDPM   IDDPM   PNDM    ProGAN  ProjectedGAN  StyleGAN  Diff-ProjectedGAN   Diff-StyleGAN2

python blip2_detect_aligned.py \
    --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
    --base_model ./blip2-opt-2.7b \
    --epochs 20 \
    --batch_size 32 \
    --save_path ./SaveFineTune/LDM-train-aligned--epochs0020

"""

import os
import argparse
import time

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


# -----------------------
# 一些固定设置
# -----------------------
PROMPT_TEXT = "Is this image fake or real? Answer ONLY with 'fake' or 'real'(no extra words)."

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# 数据集定义
# -----------------------
class FakeRealDataset(Dataset):
    """
    读取 CSV（至少包含 image, label 列）：
      - image: 图像路径
      - label: 0/1 (0=real, 1=fake)

    训练时我们不在 __getitem__ 里做 tokenizer，
    而是在 collate_fn 里一次性处理一个 batch。
    """
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns:
            raise ValueError("CSV 中缺少 'image' 列")
        if "label" not in self.df.columns:
            raise ValueError("CSV 中缺少 'label' 列 (0/1)")

        # 保证 label 是 int
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image"]
        label = int(row["label"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像: {image_path}")

        return {
            "image_path": image_path,
            "label": label,
        }


# def collate_fn(batch, processor, device):
#     """
#     把一个 batch 的样本打包成模型需要的输入：
#       - 输入 text: 同一个 PROMPT_TEXT
#       - 输入 image: 对应的图像
#       - labels: 只在最后的 "fake" 或 "real" token 上计算 loss
#     """
#     image_paths = [item["image_path"] for item in batch]
#     labels_01 = torch.tensor([item["label"] for item in batch], dtype=torch.long)
#
#     # 打开图片
#     images = [Image.open(p).convert("RGB") for p in image_paths]
#
#     # 把 0/1 转成文本答案
#     answers = ["fake" if l.item() == 1 else "real" for l in labels_01]
#
#     # 1) 构造 输入：图片 + (prompt + 答案)
#     #    训练时给模型看完整的 "prompt + answer" 序列，
#     #    但等会只在 answer 那部分算 loss
#     texts = [f"{PROMPT_TEXT} {ans}" for ans in answers]
#
#     inputs = processor(
#         images=images,
#         text=texts,
#         return_tensors="pt",
#         padding=True
#     )
#
#     # 注意：processor 默认返回的是 CPU tensor，再手动搬到 device
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     pixel_values = inputs["pixel_values"]
#
#     # 2) 构造 labels：同 shape 的张量，先全部填 -100（忽略）
#     labels_ids = torch.full_like(input_ids, fill_value=-100)
#
#     # 3) 把答案 token 放到 labels 的最后几位（对齐到右侧）
#     answer_tokens = processor.tokenizer(
#         answers,
#         return_tensors="pt",
#         padding=True
#     ).input_ids  # [bs, L_ans]
#
#     # 假设答案 token 数不会超过 input_ids 的长度
#     ans_len = answer_tokens.shape[1]
#     labels_ids[:, -ans_len:] = answer_tokens
#
#     batch_out = {
#         "input_ids": input_ids.to(device),
#         "attention_mask": attention_mask.to(device),
#         "pixel_values": pixel_values.to(device, dtype=torch.float16),
#         "labels": labels_ids.to(device),
#         # 方便将来想算分类精度的话可以用
#         "cls_labels": labels_01.to(device),
#     }
#     return batch_out
def collate_fn(batch, processor):
    """
    只在 CPU 上做打包，不要在这里 .to(cuda)
    """
    image_paths = [item["image_path"] for item in batch]
    labels_01 = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    answers = ["fake" if l.item() == 1 else "real" for l in labels_01]
    texts = [f"{PROMPT_TEXT} {ans}" for ans in answers]

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    labels_ids = torch.full_like(input_ids, fill_value=-100)
    answer_tokens = processor.tokenizer(
        answers,
        return_tensors="pt",
        padding=True
    ).input_ids
    ans_len = answer_tokens.shape[1]
    labels_ids[:, -ans_len:] = answer_tokens

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels_ids,
        "cls_labels": labels_01,
    }
    return batch_out





# -----------------------
# 构造 LoRA 模型
# -----------------------
def build_model_and_processor(base_model_path: str, lora_r: int, lora_alpha: int,
                              lora_dropout: float, device: str):
    print(f"[INFO] 加载基础模型: {base_model_path}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16
    )

    # LoRA 配置：常见选择是对 q_proj / v_proj 打 LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)

    return model, processor


# -----------------------
# 训练主函数
# -----------------------
def train(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    # 1. 模型 & processor
    model, processor = build_model_and_processor(
        opt.base_model,
        opt.lora_r,
        opt.lora_alpha,
        opt.lora_dropout,
        device,
    )

    # 2. 数据
    # dataset = FakeRealDataset(opt.dataset)
    # collate = lambda batch: collate_fn(batch, processor, device)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     collate_fn=collate,
    #     pin_memory=True
    # )
    dataset = FakeRealDataset(opt.dataset)
    collate = lambda batch: collate_fn(batch, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    model.train()

    os.makedirs(opt.save_path, exist_ok=True)

    print(f"[INFO] 数据集大小: {len(dataset)} 样本, batch_size={opt.batch_size}, "
          f"每个 epoch {len(dataloader)} 个 step")
    print(f"[INFO] 训练 prompt: \"{PROMPT_TEXT}\"")

    # 4. 训练循环
    for epoch in range(opt.epochs):
        print(f"\n========== Epoch [{epoch + 1}/{opt.epochs}] ==========")
        epoch_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{opt.epochs}"
        )

        # for step, batch in progress_bar:
        #     outputs = model(
        #         input_ids=batch["input_ids"],
        #         attention_mask=batch["attention_mask"],
        #         pixel_values=batch["pixel_values"],
        #         labels=batch["labels"],
        #     )
        for step, batch in progress_bar:
            # 这里在主进程里搬到 cuda
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)
            cls_labels = batch["cls_labels"].to(device)  # 如果之后要用的话
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )


            loss = outputs.loss
            loss_val = loss.item()
            epoch_loss += loss_val

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = epoch_loss / (step + 1)
            seen_imgs = (step + 1) * opt.batch_size

            progress_bar.set_postfix({
                "step_loss": f"{loss_val:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "imgs": f"{min(seen_imgs, len(dataset))}/{len(dataset)}"
            })

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{opt.epochs}] finished. "
              f"Avg loss: {epoch_loss / len(dataloader):.4f} "
              f"(time: {epoch_time:.1f}s)")

        # 每个 epoch 保存一次
        save_dir_epoch = os.path.join(opt.save_path, f"epoch{epoch + 1:03d}")
        os.makedirs(save_dir_epoch, exist_ok=True)
        print(f"[INFO] 保存 LoRA 权重到: {save_dir_epoch}")
        model.save_pretrained(save_dir_epoch)

    print("[INFO] 训练完成！最终权重保存在目录:", opt.save_path)


# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BLIP2 with LoRA for fake/real image detection (aligned with blip2_test006)."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="训练 CSV 路径，需包含 'image', 'label' 列（label=0/1）"
    )
    parser.add_argument(
        "--base_model", type=str,
        default="./blip2-opt-2.7b",
        help="BLIP2 基础模型路径（与测试脚本保持一致）"
    )
    parser.add_argument(
        "--save_path", type=str,
        default="./SaveFineTune/LDM-train-epochs-aligned",
        help="LoRA 权重保存目录"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)  #4

    # LoRA 超参数（可以按需调整，和你之前脚本保持一致也行）
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
