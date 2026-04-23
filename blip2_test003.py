import os
import PIL
import pandas as pd
import numpy as np
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from peft import PeftModel
import matplotlib.pyplot as plt
import tqdm
import time
import argparse


# =============== 工具函数 ===============
def map_text_to_binary(text):
    """将模型输出映射为二进制标签"""
    text_lower = text.lower().strip()
    if "fake" in text_lower:
        return 1
    elif "real" in text_lower:
        return 0
    else:
        print(f"[警告] 无法识别文本: '{text}'，默认标记为0（real）")
        return 0


def collate_fn(batch):
    """DataLoader批处理"""
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch],
                padding=True,
                return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


def evaluate_dataset(csv_path, model, processor, device, output_dir):
    """单个数据集评估流程"""
    test_df = pd.read_csv(csv_path)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n📂 [DATASET] {dataset_name} - 共 {len(test_df)} 张图像")

    test_dataset = ImageCaptioningDataset(test_df, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    result = []
    prompt = "Is this image fake or real? Answer with 'fake' or 'real'."
    start_time = time.time()

    for batch in tqdm.tqdm(test_dataloader, desc=f"Evaluating {dataset_name}"):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
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

    elapsed = time.time() - start_time
    print(f"⏱ 推理耗时: {elapsed:.2f}s")

    # ======== 结果整理 ========
    result_df = pd.DataFrame({
        "image": test_df["image"],
        "GT": test_df["text"],
        "Tlabel": test_df["text"].apply(map_text_to_binary),
        "Pred": result,
        "Plabel": [map_text_to_binary(x) for x in result],
    })

    # 检查预测结果是否全部相同
    if len(result_df["Plabel"].unique()) == 1:
        print(f"⚠️ [警告] {dataset_name}: 所有预测结果相同 ({result_df['Plabel'].unique()[0]})！")

    # ======== 指标计算 ========
    y_true, y_pred = result_df["Tlabel"], result_df["Plabel"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    print(f"📊 {dataset_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

    # ======== 保存结果 ========
    result_csv_path = os.path.join(output_dir, f"result_{dataset_name}.csv")
    result_df.to_csv(result_csv_path, index=False)
    print(f"💾 已保存详细结果: {result_csv_path}")

    return {
        "Dataset": dataset_name,
        "NumSamples": len(test_df),
        "Accuracy": acc,
        "F1": f1,
        "AUC": auc_score,
        "Duration(s)": elapsed
    }


# =============== 主函数入口 ===============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-dataset evaluation for BLIP2 + Bi-LORA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model weights")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="List of CSV paths for evaluation")
    parser.add_argument("--base_model", type=str, default="/root/autodl-tmp/project/VLM-DETECT-main/blip2-opt-2.7b")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save evaluation outputs")
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading base model from: {opt.base_model}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        opt.base_model, load_in_8bit=True, device_map="auto"
    )
    print(f"[INFO] Loading fine-tuned PEFT weights from: {opt.model_path}")
    model = PeftModel.from_pretrained(model, opt.model_path)
    processor = AutoProcessor.from_pretrained(opt.base_model, use_fast=True)

    all_results = []

    # ======== 遍历多个数据集 ========
    for csv_path in opt.datasets:
        if not os.path.exists(csv_path):
            print(f"❌ 跳过不存在的数据集: {csv_path}")
            continue

        res = evaluate_dataset(csv_path, model, processor, device, opt.output_dir)
        all_results.append(res)

    # ======== 汇总结果 ========
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(opt.output_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n✅ 所有数据集评估完成！结果汇总：")
    print(summary_df)
    print(f"\n💾 汇总结果已保存至: {summary_path}")

#加入多测试集自动评估（可一次性跑多个CSV并汇总成表）
#使用方法：
# python blip2_test003.py \
#     --model_path /root/autodl-tmp/project/VLM-DETECT-main/SaveFineTune \
#     --datasets \
#         /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_ADM.csv \
#         /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_DDPM.csv \
#         /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_LDM.csv \
#     --output_dir ./multi_eval_results

#输出说明，每个测试集单独保存：
# result_test_adm.csv
# result_test_ddpm.csv
# result_test_ldm.csv


