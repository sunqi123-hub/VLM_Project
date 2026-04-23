"""
1,新增 SRM 和频域编码器。
2,自定义 MoE-LoRA 层：手动替换 LLM 的线性层，而不是使用 peft 库（因为 peft 暂不支持动态路由 MoE）。
将原本的单一任务模型转变为一个具备**多视角感知（RGB+Freq）和动态适应能力（MoE）**的强大检测器---gemini生成

python blip2_detect-aligned01.py \
  --dataset ./data/Train_CSV/train_LDM.csv \
  --base_model ./blip2-opt-2.7b \
  --save_path ./Save_MoE_CoT/LDM-train-aligned01-epochs0010 \
  --epochs 10 \
  --batch_size 4 \
  --num_experts 3

"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# 允许加载截断图片
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------
# [保持不变] 1. 频域感知模块 (Artifact-Aware Branch)
# -----------------------
class SRMConv(nn.Module):
    """固定权重的 SRM 滤波器，用于提取噪声残差"""

    def __init__(self):
        super().__init__()
        self.channels = 3
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        filter1 = np.array(filter1, dtype=float) / q[0]
        filter2 = np.array(filter2, dtype=float) / q[1]
        filter3 = np.array(filter3, dtype=float) / q[2]

        filters = np.array([[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])
        weight = torch.tensor(filters, dtype=torch.float32)
        self.register_buffer('weight', weight, persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=1, padding=2)


class FrequencyEncoder(nn.Module):
    """轻量级 CNN，将 SRM 特征编码为向量"""

    def __init__(self, output_dim=2560):
        super().__init__()
        self.srm = SRMConv()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, x):
        # x 期望是 float32
        noise = self.srm(x)
        feat = self.net(noise)
        return self.projection(feat).unsqueeze(1)  # [B, 1, Dim]


# -----------------------
# [保持不变] 2. MoE-LoRA 模块 (Dynamic Adapter)
# -----------------------
class MoELoraLinear(nn.Module):
    def __init__(self, original_layer, r=16, num_experts=3, alpha=32, dropout=0.05, device=None):
        super().__init__()
        # self.base_layer = original_layer
        # self.in_features = original_layer.in_features
        # self.out_features = original_layer.out_features
        # self.r = r
        # self.num_experts = num_experts
        # self.scaling = alpha / r
        #
        # # Router
        # self.router = nn.Linear(self.in_features, num_experts)
        #
        # # Experts
        # self.lora_A = nn.ParameterList(
        #     [nn.Parameter(torch.randn(self.r, self.in_features)) for _ in range(num_experts)])
        # self.lora_B = nn.ParameterList(
        #     [nn.Parameter(torch.zeros(self.out_features, self.r)) for _ in range(num_experts)])
        # self.dropout = nn.Dropout(dropout)
        self.base_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = r
        self.num_experts = num_experts
        self.scaling = alpha / r
        # Router - 修复点 1：创建时指定设备
        self.router = nn.Linear(self.in_features, num_experts, device=device)
        # Experts - 修复点 2：创建时指定设备
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.randn(self.r, self.in_features, device=device)) for _ in range(num_experts)])
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.out_features, self.r, device=device)) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)

        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=5 ** 0.5)
            nn.init.zeros_(self.lora_B[i])

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        result = self.base_layer(x)
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1)  # [B, S, Num_Experts]

        moe_out = 0
        x_drop = self.dropout(x)

        for i in range(self.num_experts):
            expert_val = (x_drop @ self.lora_A[i].T) @ self.lora_B[i].T
            expert_val = expert_val * self.scaling
            weight_i = router_weights[:, :, i].unsqueeze(-1)
            moe_out += expert_val * weight_i

        return result + moe_out


# -----------------------
# [保持不变] 3. 模型包装器 (Model Wrapper)
# -----------------------
class ArtifactMoEBlip2(nn.Module):
    def __init__(self, base_model_path, device, num_experts=3):
        super().__init__()
        print(f"[INFO] 加载基础模型: {base_model_path}")
        self.base_model = Blip2ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        # self.inject_moe_lora(self.base_model.language_model, num_experts)
        self.inject_moe_lora(self.base_model.language_model, num_experts, device)

        hidden_size = self.base_model.language_model.config.hidden_size
        self.freq_encoder = FrequencyEncoder(output_dim=hidden_size).to(device)

    # def inject_moe_lora(self, model, num_experts):
    #     print("[INFO] 正在注入 MoE-LoRA 层...")
    #     count = 0
    #     for name, module in model.named_modules():
    #         if "q_proj" in name or "v_proj" in name:
    #             parent_name = name.rsplit('.', 1)[0]
    #             child_name = name.rsplit('.', 1)[1]
    #             parent = model.get_submodule(parent_name)
    #
    #             original_linear = getattr(parent, child_name)
    #             moe_layer = MoELoraLinear(original_linear, num_experts=num_experts)
    #
    #             setattr(parent, child_name, moe_layer)
    #             count += 1
    #     print(f"[INFO] 已替换 {count} 个层为 MoE-LoRA 专家层。")
        # 修复后的代码
        def inject_moe_lora(self, model, num_experts, device):  # 修复点 3a: 接受 device
            print("[INFO] 正在注入 MoE-LoRA 层...")
            count = 0
            for name, module in model.named_modules():
                if "q_proj" in name or "v_proj" in name:
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.rsplit('.', 1)[1]
                    parent = model.get_submodule(parent_name)
                    original_linear = getattr(parent, child_name)
                    moe_layer = MoELoraLinear(original_linear, num_experts=num_experts,
                                              device=device)  # 修复点 3b: 传递 device
                    setattr(parent, child_name, moe_layer)
                    count += 1
            print(f"[INFO] 已替换 {count} 个层为 MoE-LoRA 专家层。")


    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.base_model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        qformer_outputs = self.base_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = qformer_outputs.last_hidden_state
        # === 修复：确保 Q-Former 输出与投影层权重精度一致 ===改
        target_dtype = self.base_model.language_projection.weight.dtype
        if query_output.dtype != target_dtype:
            query_output = query_output.to(target_dtype)
        # =======================================================
        language_model_inputs = self.base_model.language_projection(query_output)

        # 频域特征融合
        freq_embeds = self.freq_encoder(pixel_values.float())
        if language_model_inputs.dtype == torch.float16:
            freq_embeds = freq_embeds.half()

        inputs_embeds = torch.cat([language_model_inputs, freq_embeds], dim=1)
        visual_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=inputs_embeds.device)

        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)

        final_inputs = torch.cat([inputs_embeds, text_embeds], dim=1)
        final_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full((inputs_embeds.shape[0], inputs_embeds.shape[1]), -100, device=labels.device)
            final_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            final_labels = None

        return self.base_model.language_model(
            inputs_embeds=final_inputs,
            attention_mask=final_mask,
            labels=final_labels,
            return_dict=True
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        moe_state_dict = {}
        for name, param in self.named_parameters():
            if "router" in name or "lora_" in name:
                moe_state_dict[name] = param.cpu()
        torch.save(moe_state_dict, os.path.join(save_directory, "moe_lora_weights.pt"))
        torch.save(self.freq_encoder.state_dict(), os.path.join(save_directory, "freq_encoder.pt"))
        print(f"[INFO] 模型权重已保存至 {save_directory}")


# -----------------------
# [修改部分] 4. 数据处理 (回归标准二分类)
# -----------------------
PROMPT_TEXT = "Is this image fake or real? Answer ONLY with 'fake' or 'real'(no extra words)."


class FakeRealDataset(Dataset):
    """
    标准数据集加载类
    只读取 image 路径 和 label (0/1)
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV 必须包含 'image' 和 'label' 列")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image"]
        label_int = int(row["label"])  # 0=real, 1=fake

        # 统一转为文本标签
        answer = "fake" if label_int == 1 else "real"

        return {"image_path": img_path, "answer": answer}


def collate_fn_advanced(batch, processor):
    image_paths = [x["image_path"] for x in batch]
    answers = [x["answer"] for x in batch]

    # 构造输入文本: Prompt + Answer
    texts = [f"{PROMPT_TEXT} {ans}" for ans in answers]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # Tokenize
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    pixel_values = inputs.pixel_values

    # Label 构造: 忽略掉 prompt 部分，只在 answer 部分计算 loss
    # 先全部设为 -100
    labels = torch.full_like(input_ids, fill_value=-100)

    # 单独对 answer 进行 tokenize，以便知道它的长度
    answer_tokens = processor.tokenizer(answers, return_tensors="pt", padding=True).input_ids

    # 将 answer 的 token 填入 labels 的尾部 (假设是右对齐)
    # 这里的简化逻辑是：取 input_ids 的最后几个 token 作为 answer
    ans_len = answer_tokens.shape[1]
    labels[:, -ans_len:] = input_ids[:, -ans_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels
    }


# -----------------------
# 5. 训练循环 (Main)
# -----------------------
accumulation_steps = 8 #后加 改
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化自定义模型 (包含 MoE 和 FreqEncoder)
    model = ArtifactMoEBlip2(args.base_model, device, num_experts=args.num_experts)
    processor = AutoProcessor.from_pretrained(args.base_model)

    # 打印可训练参数
    trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"[INFO] 可训练参数量: {trainable_params / 1e6:.2f} M")

    # 使用修改后的 Dataset 和 Collate 函数
    dataset = FakeRealDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn_advanced(b, processor),
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.train()

    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        total_loss = 0

        for batch in loop:
            optimizer.zero_grad()

            # 数据移至 GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss
            #loss.backward()  #替换
            loss = outputs.loss / accumulation_steps  # 归一化 Loss
            loss.backward()
            # 每隔 accumulation_steps 步才执行一次优化器更新
            if (idx + 1) % accumulation_steps == 0 or idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # 保存
        save_path = os.path.join(args.save_path, f"epoch_{epoch + 1}")
        model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to training CSV (must have 'image' and 'label')")
    parser.add_argument("--base_model", default="./blip2-opt-2.7b")
    parser.add_argument("--save_path", default="./Save_MoE_CoT/LDM-train-aligned01-epochs0010")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_experts", type=int, default=3)
    args = parser.parse_args()

    train(args)