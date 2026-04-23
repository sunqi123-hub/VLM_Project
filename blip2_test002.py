import os
import PIL
import pandas as pd
import numpy as np
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
import tqdm
import time
import argparse

# =============== 全局随机种子 ===============
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============== 工具函数 ===============
def map_text_to_binary(text):
    """将模型输出映射为二进制标签"""
    text_lower = text.lower().strip()
    if "fake" in text_lower:
        return 1
    elif "real" in text_lower:
        return 0
    else:
        print(f"[警告] 无法识别文本: '{text}'，已标记为0（real）")
        return 0


def collate_fn(batch):
    """数据加载时批处理函数"""
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


# =============== 主函数 ===============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Fine-Tuned BLIP-2 with Bi-LORA for Synthetic Image Detection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model weights (PEFT directory)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test CSV file (with image paths & labels)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/root/autodl-tmp/project/VLM-DETECT-main/blip2-opt-2.7b",
        help="Path to BLIP2 base model",
    )
    opt = parser.parse_args()

    # =============== 加载模型与处理器 ===============
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading base model from: {opt.base_model}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        opt.base_model, load_in_8bit=True, device_map="auto"
    )
    print(f"[INFO] Loading fine-tuned weights from: {opt.model_path}")
    model = PeftModel.from_pretrained(model, opt.model_path)
    processor = AutoProcessor.from_pretrained(opt.base_model, use_fast=True)

    # =============== 加载测试数据 ===============
    test_df = pd.read_csv(opt.dataset)
    test_df = test_df.sample(n=200, random_state=42).reset_index(drop=True)  #随机选取200个
    # # 提取前100张和后100张样本（确保数据集总行数 >= 200，否则tail(100)会取实际剩余数量）
    # # 前100张
    # test_df_head = test_df.head(300)
    # # 后100张
    # test_df_tail = test_df.tail(700)
    # # 合并为新的测试集（共200张），并重置索引
    # test_df = pd.concat([test_df_head, test_df_tail], ignore_index=True)

    print(f"[INFO] Loaded {len(test_df)} samples from {opt.dataset}")

    test_dataset = ImageCaptioningDataset(test_df, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    # =============== 推理 ===============
    result = []
    start_time = time.time()

    prompt = "Is this image fake or real? Answer with 'fake' or 'real'."

    for batch in tqdm.tqdm(test_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        # prompt嵌入
        prompt_inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = prompt_inputs["input_ids"]

        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=10,
            num_beams=5,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True
        )

        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result.append(pred_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Inference completed in {elapsed_time:.2f} seconds")

    # =============== 结果整理 ===============
    result_df = pd.DataFrame({
        "image": test_df["image"],
        "GT": test_df["text"],
        "Tlabel": test_df["text"].apply(map_text_to_binary),
        "Pred": result,
        "Plabel": [map_text_to_binary(x) for x in result],
    })

    # =============== 统计与指标计算 ===============
    print("\n[INFO] Prediction distribution:")
    print(result_df["Pred"].value_counts())
    print(result_df["Plabel"].value_counts())

    # 检查预测结果是否全部相同
    if len(result_df["Plabel"].unique()) == 1:
        print("[警告] 所有预测结果完全相同！模型可能输出固定值，请检查权重或输入。")

    # 正确顺序的指标计算
    accuracy = accuracy_score(result_df["Tlabel"], result_df["Plabel"])
    f1Score = f1_score(result_df["Tlabel"], result_df["Plabel"])
    fpr, tpr, th = roc_curve(result_df["Tlabel"], result_df["Plabel"])
    auc_score = auc(fpr, tpr)

    print("\n=== METRICS ===")
    print(f"AUC: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1Score:.4f}")
    print("================")

    # 输出前几个样本以检查生成结果
    print("\n[Sample Predictions]")
    print(result_df.head(10))

    # 保存结果
    save_path = os.path.join(os.path.dirname(opt.dataset), "result_eval.csv")
    result_df.to_csv(save_path, index=False)
    print(f"[INFO] Saved detailed results to {save_path}")


# ✅ 自动打印预测结果分布，检测“全输出相同”的异常；
# ✅ 修复 F1 和 Accuracy 顺序错误；
# ✅ 增加样本输出便于人工核查；
# ✅ 引入温度采样避免生成固定答案；
# ✅ 输出评估CSV文件，便于后续分析。