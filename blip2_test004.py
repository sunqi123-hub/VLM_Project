"""
 和blip2_test005.py一样
 ./SaveFineTune/LDM-train-all-aligned-epochs00020/epoch020              ./SaveFineTune/LDM-train-aligned-epochs0020/epoch020 \
./SaveFineTune/LDM-train-aligned--epochs000020   sample5000-aligned--00020-In_LDM_Results参数和论文里一样./Test-Results/In_LDM_Results/sample5000-aligned--00020-In_LDM_Results   aligned--00020-In_LDM_Results
sample500-all-aligned-020-In_LDM_Results   sample500-all-aligned-010-In_LDM_Results      LDM-train-all-aligned-epochs00020  --batch_size 64


python blip2_test004.py \
    --model_path  ./SaveFineTune/DDPM-train-aligned--epochs020/epoch020 \
    --dataset ./data/Test_CSV/test_LDM.csv \
              ./data/Test_CSV/test_ADM.csv \
              ./data/Test_CSV/test_DDPM.csv \
              ./data/Test_CSV/test_IDDPM.csv \
              ./data/Test_CSV/test_PNDM.csv \
    --num_samples 500

python blip2_test004.py \
    --model_path  ./SaveFineTune/PNDM-train-aligned--epochs020/epoch020 \
    --dataset ./data/Test_CSV/test_StyleGAN.csv \
              ./data/Test_CSV/test_Diff-StyleGAN2.csv \
              ./data/Test_CSV/test_Diff-ProjectedGAN.csv \
              ./data/Test_CSV/test_ProGAN.csv \
              ./data/Test_CSV/test_ProjectedGAN.csv \
    --num_samples 500

python blip2_test004.py \
  --base_model ./blip2-opt-2.7b \
  --model_path ./SaveFineTune/LDM-gnn-cot_test/epoch020 \
  --gnn_head_path ./SaveFineTune/LDM-gnn-cot_test/epoch020/gnn_cot_head.pt \
  --dataset ./data/Test_CSV/test_LDM.csv \
  --structured_cot \
  --num_samples 500 \
  --decision_source lm

#自己训练--model_path    ./SaveFineTune/LDM-train-epochs03
改63行
和6的基础上运行时去掉字体提示
测试多个数据集示例    ADM  DDPM  Diff-ProjectedGAN   Diff-StyleGAN2  IDDPM   LDM  PNDM  ProGAN  ProjectedGAN  StyleGAN
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from peft import PeftModel
import tqdm
import time
import argparse
import re
from PIL import Image
import matplotlib.font_manager as fm  # 添加这行导入语句
from gnn_cot import format_top_patches, load_gnn_cot_head

import warnings  # 后加
import logging
# 1️⃣ 彻底屏蔽 matplotlib 的字体查找提示
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# 2️⃣ 屏蔽所有字体相关的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===========================
# 设置随机种子，保证可复现性
# ===========================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================
# 定义目标保存目录（核心修改点1）
# ===========================
TARGET_SAVE_DIR = "./Test-Results/gnn-cot_test/gnn-cot_test02-20_500_fusion"     # In_ADM_Results/In_ADM_Results/sample500-In_ADM_Results   sample500-   all-In_LDM_Results   sample5000-aligned--00020-In_LDM_Results
# 确保目标目录存在，不存在则创建
os.makedirs(TARGET_SAVE_DIR, exist_ok=True)

PROMPT_TEXT = "Is this image fake or real? Answer ONLY with 'fake' or 'real' (no extra words)."
STRUCTURED_COT_PROMPT = (
    "You are a forensic image analyst. Decide whether the image is fake or real. "
    "Return exactly four lines with these fields: Quick intuition, Salient evidence, "
    "Deep reasoning, Final conclusion."
)

COT_TEMPLATES = {
    0: (
        "Quick intuition: real.\n"
        "Salient evidence: no localized manipulation cue is dominant in the visible regions.\n"
        "Deep reasoning: the spatial content and frequency texture are mutually consistent, "
        "so the forensic evidence supports an authentic image.\n"
        "Final conclusion: real."
    ),
    1: (
        "Quick intuition: fake.\n"
        "Salient evidence: localized visual artifacts and frequency inconsistency are suspicious.\n"
        "Deep reasoning: the spatial content and frequency texture are not fully aligned, "
        "so the forensic evidence supports a synthetic or manipulated image.\n"
        "Final conclusion: fake."
    ),
}


# ===========================
# 辅助函数定义
# ===========================
def map_text_to_binary(text):
    """将模型输出文本映射为二进制标签"""
    text_lower = text.lower().strip()
    # 移除引号、空格、重复词
    text_clean = text_lower.replace("'", "").replace(" ", "")
    # 取前4个字符（"fake"长度为4）避免重复
    text_clean = text_clean[:4]
    if "fake" in text_lower:
        return 1
    elif "real" in text_lower:
        return 0
    else:
        print(f"[警告] 无法识别文本: '{text}'，默认标记为0（real）")
        return 0


def extract_final_label(text):
    """Prefer the explicit final conclusion in a structured CoT report."""
    text_lower = str(text).lower().strip()
    final_match = re.search(r"final\s*(?:conclusion|answer)?\s*[:：]\s*(fake|real)", text_lower)
    if final_match:
        return 1 if final_match.group(1) == "fake" else 0

    for line in reversed(text_lower.splitlines()):
        if "fake" in line:
            return 1
        if "real" in line:
            return 0
    return map_text_to_binary(text)


def label_to_text(label):
    return "fake" if int(label) == 1 else "real"


def move_inputs_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.float16)
        else:
            moved[key] = value.to(device=device)
    return moved


def find_subsequence(sequence, pattern):
    if len(pattern) == 0 or len(pattern) > len(sequence):
        return None
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def build_labels_for_targets(input_ids, attention_mask, target_texts, tokenizer):
    labels = torch.full_like(input_ids, fill_value=-100)
    for row_idx, target_text in enumerate(target_texts):
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        valid_positions = attention_mask[row_idx].nonzero(as_tuple=False).flatten()
        target_len = min(len(target_ids), int(valid_positions.numel()))
        if target_len == 0:
            continue
        valid_token_ids = input_ids[row_idx, valid_positions].tolist()
        match_start = find_subsequence(valid_token_ids, target_ids[:target_len])
        if match_start is not None:
            target_positions = valid_positions[match_start : match_start + target_len]
        else:
            target_positions = valid_positions[-target_len:]
        labels[row_idx, target_positions] = input_ids[row_idx, target_positions]
    return labels


def class_targets_for_scoring(structured_cot):
    if structured_cot:
        return ["\n" + COT_TEMPLATES[0], "\n" + COT_TEMPLATES[1]]
    return [" real", " fake"]


def lm_fake_probability(model, processor, image, device, score_mode):
    use_cot_score = score_mode == "cot"
    prompt = STRUCTURED_COT_PROMPT if use_cot_score else PROMPT_TEXT
    target_texts = class_targets_for_scoring(use_cot_score)
    losses = []
    for target_text in target_texts:
        text = prompt + target_text
        inputs = processor(images=image, text=text, return_tensors="pt")
        labels = build_labels_for_targets(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            target_texts=[target_text],
            tokenizer=processor.tokenizer,
        )
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=labels,
        )
        losses.append(outputs.loss.detach().float())

    nll = torch.stack(losses)
    probs = torch.softmax(-nll, dim=0)
    return probs[1].item()


def resolve_lora_checkpoint(model_path):
    if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
        return model_path

    if os.path.isdir(model_path):
        epoch_dirs = []
        for name in os.listdir(model_path):
            candidate = os.path.join(model_path, name)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
                epoch_dirs.append(candidate)
        if epoch_dirs:
            return sorted(epoch_dirs)[-1]

    search_root = model_path
    while search_root and not os.path.isdir(search_root):
        parent = os.path.dirname(search_root)
        if parent == search_root:
            break
        search_root = parent

    nearby = []
    if search_root and os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            if "adapter_config.json" in files:
                nearby.append(root)
                if len(nearby) >= 10:
                    break

    message = f"Cannot find adapter_config.json at {model_path}."
    if nearby:
        message += "\nAvailable LoRA checkpoints nearby:\n" + "\n".join(f"  - {path}" for path in nearby)
    raise FileNotFoundError(message)


def resolve_gnn_head_path(gnn_head_path, model_path):
    if gnn_head_path is None:
        candidate = os.path.join(model_path, "gnn_cot_head.pt")
        return candidate if os.path.isfile(candidate) else None
    if os.path.isfile(gnn_head_path):
        return gnn_head_path
    fallback = os.path.join(model_path, "gnn_cot_head.pt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Cannot find GNN head at {gnn_head_path} or {fallback}.")


def _metric_value(y_true, y_pred, metric_name):
    if metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred)


def find_best_threshold(y_true, y_score, metric_name="f1"):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_score.size == 0:
        return 0.5, 0.0

    unique_scores = np.unique(y_score)
    candidates = [float(unique_scores[0]) - 1e-12, float(unique_scores[-1]) + 1e-12]
    candidates.extend(float(score) for score in unique_scores)
    if unique_scores.size > 1:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.extend(float(score) for score in mids)

    best_threshold = 0.5
    best_metric = -1.0
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(int)
        metric = _metric_value(y_true, y_pred, metric_name)
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold
    return best_threshold, best_metric


def find_best_fusion_weight(y_true, lm_score, gnn_score):
    y_true = np.asarray(y_true, dtype=int)
    lm_score = np.asarray(lm_score, dtype=float)
    gnn_score = np.asarray(gnn_score, dtype=float)
    valid = np.isfinite(lm_score) & np.isfinite(gnn_score)
    if not valid.any():
        return 0.0, lm_score

    best_weight = 0.0
    best_auc = -1.0
    best_score = lm_score
    for weight in np.linspace(0.0, 1.0, 21):
        score = (1.0 - weight) * lm_score + weight * gnn_score
        try:
            fpr, tpr, _ = roc_curve(y_true[valid], score[valid])
            auc_score = auc(fpr, tpr)
        except ValueError:
            auc_score = -1.0
        if auc_score > best_auc:
            best_auc = auc_score
            best_weight = float(weight)
            best_score = score
    return best_weight, best_score


# def plot_single_roc(y_true, y_pred, dataset_name, save_path):
#     """绘制单个数据集的ROC曲线并保存"""
#     fpr, tpr, _ = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('假正例率 (FPR)')
#     plt.ylabel('真正例率 (TPR)')
#     plt.title(f'{dataset_name} 的ROC曲线')
#     plt.legend(loc="lower right")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     return fpr, tpr, roc_auc


def plot_single_roc(y_true, y_pred, dataset_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    # 直接指定字体文件
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    font_prop = fm.FontProperties(fname=font_path)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)', fontproperties=font_prop)  # 每个文本单独指定字体
    plt.ylabel('真正例率 (TPR)', fontproperties=font_prop)
    plt.title(f'{dataset_name} 的ROC曲线', fontproperties=font_prop)
    plt.legend(loc="lower right", prop=font_prop)  # 图例字体
    # plt.xlabel('假正例率 (FPR)' if font_prop else 'FPR',
    #            fontproperties=font_prop if font_prop else None)
    # plt.ylabel('真正例率 (TPR)' if font_prop else 'TPR',
    #            fontproperties=font_prop if font_prop else None)
    # plt.title(f'{dataset_name} 的ROC曲线' if font_prop else f'{dataset_name} ROC Curve',
    #           fontproperties=font_prop if font_prop else None)
    # plt.legend(loc="lower right", prop=font_prop if font_prop else None)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fpr, tpr, roc_auc


def plot_combined_roc(all_results, save_path):
    """绘制所有数据集的汇总ROC曲线"""
    plt.figure(figsize=(10, 8))

    for result in all_results:
        plt.plot(
            result['fpr'],
            result['tpr'],
            lw=2,
            label=f'{result["name"]} (AUC = {result["auc"]:.4f})'
        )

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('各数据集ROC曲线对比')
    plt.legend(loc="lower right")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_performance_table(all_results, save_path):
    """生成性能指标对比表格图像"""
    # 准备表格数据
    metrics = []
    for result in all_results:
        metrics.append([
            result['name'],
            f"{result['auc']:.4f}",
            f"{result['accuracy']:.4f}",
            f"{result['f1']:.4f}"
        ])

    # 创建表格
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=metrics,
        colLabels=['数据集', 'AUC', '准确率', 'F1分数'],
        cellLoc='center',
        loc='center'
    )

    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title('各数据集性能指标对比', fontsize=14, pad=20)
    # plt.title('各数据集性能指标对比' if font_prop else 'Performance Comparison',
    #           fontsize=14, pad=20, fontproperties=font_prop if font_prop else None)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_dataset(dataset_path, model, processor, device, opt, gnn_head=None):
    """处理单个数据集并返回结果"""
    # 提取数据集名称
    dataset_name = os.path.basename(dataset_path).split('.')[0].replace('test_', '')

    # 加载测试集
    test_df = pd.read_csv(dataset_path)
    if opt.num_samples is not None:
        test_df = test_df.sample(n=opt.num_samples, random_state=42).reset_index(drop=True)
        print(f"[INFO] 从 {dataset_path} 随机选择了 {len(test_df)} 个样本")
    else:
        print(f"[INFO] 从 {dataset_path} 加载了 {len(test_df)} 个样本")

    # 推理阶段
    results = []
    text_results = []
    lm_fake_probs = []
    cot_reports = []
    gnn_fake_probs = []
    fake_scores = []
    gnn_evidence = []
    start_time = time.time()
    prompt_text = STRUCTURED_COT_PROMPT if opt.structured_cot else PROMPT_TEXT
    max_new_tokens = opt.max_new_tokens if opt.structured_cot else 1

    for i, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df), desc=f"Testing {dataset_name}"):
        image_path = row["image"]
        if not os.path.exists(image_path):
            print(f"[跳过] 找不到图像: {image_path}")
            results.append("error")
            text_results.append("error")
            lm_fake_probs.append(np.nan)
            cot_reports.append("missing image")
            gnn_fake_probs.append(np.nan)
            fake_scores.append(np.nan)
            gnn_evidence.append("")
            continue

        try:
            # 处理输入
            image = Image.open(image_path).convert("RGB")
            inputs = move_inputs_to_device(processor(
                images=image,
                text=prompt_text,
                return_tensors="pt"
            ), device)

            # 执行推理
            with torch.no_grad():
                lm_prob_fake = lm_fake_probability(
                    model=model,
                    processor=processor,
                    image=image,
                    device=device,
                    score_mode=opt.score_mode,
                )
                pred_text = ""
                if opt.generate_reports or opt.decision_source == "generate":
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                        do_sample=False
                    )
                    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    pred_text = pred_text.replace(prompt_text, "").strip()
                graph_prob_fake = np.nan
                graph_label = None
                evidence_text = ""
                if gnn_head is not None:
                    graph_outputs = gnn_head(
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    graph_prob_fake = graph_outputs["logits"].softmax(dim=-1)[0, 1].item()
                    graph_label = int(graph_prob_fake >= 0.5)
                    evidence = format_top_patches(graph_outputs, top_k=opt.gnn_top_k)
                    evidence_text = evidence[0] if evidence else ""

            # 解码结果
            generated_label = None
            if pred_text:
                generated_label = extract_final_label(pred_text) if opt.structured_cot else map_text_to_binary(pred_text)

            lm_label = int(lm_prob_fake >= opt.threshold)
            text_label = generated_label if opt.decision_source == "generate" and generated_label is not None else lm_label

            if opt.decision_source == "gnn" and graph_label is not None:
                final_score = graph_prob_fake
            elif opt.decision_source == "fusion" and graph_label is not None:
                final_score = opt.gnn_vote_weight * graph_prob_fake + (1.0 - opt.gnn_vote_weight) * lm_prob_fake
            elif opt.decision_source == "generate" and generated_label is not None:
                final_score = float(generated_label)
            else:
                final_score = lm_prob_fake
            final_label = int(final_score >= opt.threshold)

            results.append(label_to_text(final_label))
            text_results.append(label_to_text(text_label))
            lm_fake_probs.append(lm_prob_fake)
            cot_reports.append(pred_text)
            gnn_fake_probs.append(graph_prob_fake)
            fake_scores.append(final_score)
            gnn_evidence.append(evidence_text)

        except Exception as e:
            print(f"[错误] 图像 {image_path} 推理失败: {e}")
            results.append("error")
            text_results.append("error")
            lm_fake_probs.append(np.nan)
            cot_reports.append(f"error: {e}")
            gnn_fake_probs.append(np.nan)
            fake_scores.append(np.nan)
            gnn_evidence.append("")

    end_time = time.time()
    print(f"[INFO] {dataset_name} 推理完成，耗时 {end_time - start_time:.2f} 秒")

    # 结果整理
    result_df = pd.DataFrame({
        "image": test_df["image"],
        "GT": test_df["text"],
        "Tlabel": test_df["text"].apply(map_text_to_binary),
        "Pred": results,
        "TextPred": text_results,
        "LMFakeProb": lm_fake_probs,
        "GNNFakeProb": gnn_fake_probs,
        "FakeScore": fake_scores,
        "DecisionSource": opt.decision_source,
        "GNNEvidence": gnn_evidence,
        "CoTReport": cot_reports,
    })

    y_true = result_df["Tlabel"].astype(int)
    y_score = result_df["FakeScore"].astype(float).fillna(0.0)

    applied_fusion_weight = opt.gnn_vote_weight
    if opt.auto_fusion_weight:
        lm_score = result_df["LMFakeProb"].astype(float).to_numpy()
        gnn_score = result_df["GNNFakeProb"].astype(float).to_numpy()
        applied_fusion_weight, fused_score = find_best_fusion_weight(y_true, lm_score, gnn_score)
        y_score = pd.Series(fused_score, index=result_df.index).fillna(0.0)
        result_df["FakeScore"] = y_score
        result_df["DecisionSource"] = f"auto_fusion:{applied_fusion_weight:.2f}"

    applied_threshold = opt.threshold
    if opt.auto_threshold:
        applied_threshold, best_metric = find_best_threshold(y_true, y_score, opt.threshold_metric)
        print(f"[INFO] 自动阈值({opt.threshold_metric})={applied_threshold:.12f}, best={best_metric:.4f}")

    y_pred = (y_score >= applied_threshold).astype(int)
    result_df["Plabel"] = y_pred
    result_df["Pred"] = result_df["Plabel"].apply(label_to_text)
    result_df["AppliedThreshold"] = applied_threshold
    result_df["AppliedFusionWeight"] = applied_fusion_weight

    # 打印预测分布
    print(f"\n[INFO] {dataset_name} 预测分布:")
    print(result_df["Pred"].value_counts())
    print(result_df["Plabel"].value_counts())

    # 检查预测一致性
    if len(result_df["Plabel"].unique()) == 1:
        print(f"[警告] {dataset_name} 所有预测结果相同！请检查模型或数据。")

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    print(f"\n=== {dataset_name} 指标 ===")
    print(f"AUC: {auc_score:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("====================")

    # 保存结果CSV（核心修改点2：改为目标目录）
    save_path = os.path.join(TARGET_SAVE_DIR, f"result_{dataset_name}.csv")
    result_df.to_csv(save_path, index=False)
    print(f"[INFO] 结果已保存至 {save_path}")

    # 绘制并保存单个ROC曲线（核心修改点3：改为目标目录）
    roc_save_path = os.path.join(TARGET_SAVE_DIR, f"roc_{dataset_name}.png")
    fpr, tpr, auc_score = plot_single_roc(y_true, y_score, dataset_name, roc_save_path)
    print(f"[INFO] ROC曲线已保存至 {roc_save_path}")

    return {
        "name": dataset_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score,
        "accuracy": accuracy,
        "f1": f1,
        "save_dir": TARGET_SAVE_DIR  # 核心修改点4：返回目标目录
    }


# ===========================
# 主函数入口
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test BLIP2 + Bi-LORA fine-tuned model on synthetic image detection"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned LoRA weights")
    parser.add_argument("--dataset", type=str, nargs='+', required=True,  # 支持多个数据集
                        help="Paths to test CSV files (multiple allowed)")
    parser.add_argument("--base_model", type=str,
                        default="/root/autodl-tmp/project/VLM-DETECT-main/blip2-opt-2.7b",
                        help="Path to BLIP2 base model")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Optional: number of samples to test (randomly sampled)")
    parser.add_argument("--structured_cot", action="store_true",
                        help="生成 quick intuition / salient evidence / deep reasoning / final conclusion 四步报告")
    parser.add_argument("--max_new_tokens", type=int, default=96,
                        help="structured_cot 模式下的最大生成 token 数")
    parser.add_argument("--gnn_head_path", type=str, default=None,
                        help="可选：加载训练保存的 gnn_cot_head.pt，用图分支融合判决")
    parser.add_argument("--gnn_vote_weight", type=float, default=0.5,
                        help="融合判决时 GNN fake 概率的权重，范围 0-1")
    parser.add_argument("--gnn_top_k", type=int, default=3,
                        help="保存 GNN 证据热区时保留的 top-k patch 数")
    parser.add_argument("--decision_source", type=str, default="lm",
                        choices=["lm", "gnn", "fusion", "generate"],
                        help="Decision source: lm uses fake/real likelihood scoring; fusion blends LM and GNN.")
    parser.add_argument("--score_mode", type=str, default="short", choices=["short", "cot"],
                        help="LM scoring target. short uses real/fake answers; cot scores full CoT templates.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fake probability threshold used for accuracy/F1.")
    parser.add_argument("--auto_threshold", action="store_true",
                        help="Sweep thresholds on the evaluated data and apply the best one.")
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "accuracy"],
                        help="Metric used by --auto_threshold.")
    parser.add_argument("--auto_fusion_weight", action="store_true",
                        help="Sweep LM/GNN fusion weights by AUC when both scores are available.")
    parser.add_argument("--generate_reports", action="store_true",
                        help="Generate CoT text reports. Classification does not depend on generated text by default.")
    opt = parser.parse_args()
    opt.gnn_vote_weight = min(max(opt.gnn_vote_weight, 0.0), 1.0)
    opt.threshold = min(max(opt.threshold, 0.0), 1.0)

    # ===========================
    # 模型加载
    # ===========================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] 加载基础模型: {opt.base_model}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        opt.base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )

    opt.model_path = resolve_lora_checkpoint(opt.model_path)
    print(f"[INFO] 加载微调LoRA权重: {opt.model_path}")
    model = PeftModel.from_pretrained(model, opt.model_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(opt.base_model, use_fast=False)
    gnn_head = None
    opt.gnn_head_path = resolve_gnn_head_path(opt.gnn_head_path, opt.model_path)
    if opt.gnn_head_path is not None:
        print(f"[INFO] 加载 GNN-CoT 头: {opt.gnn_head_path}")
        gnn_head = load_gnn_cot_head(opt.gnn_head_path, processor.tokenizer, map_location=device).to(device)
        gnn_head.eval()

    # 处理所有数据集
    all_results = []
    for dataset_path in opt.dataset:
        result = process_dataset(dataset_path, model, processor, device, opt, gnn_head=gnn_head)
        all_results.append(result)

    # 生成汇总ROC曲线（核心修改点5：改为目标目录）
    if all_results:
        combined_roc_path = os.path.join(TARGET_SAVE_DIR, "combined_roc.png")
        plot_combined_roc(all_results, combined_roc_path)
        print(f"[INFO] 汇总ROC曲线已保存至 {combined_roc_path}")

        # 生成性能表格（核心修改点6：改为目标目录）
        table_path = os.path.join(TARGET_SAVE_DIR, "performance_table.png")
        generate_performance_table(all_results, table_path)
        print(f"[INFO] 性能指标表格已保存至 {table_path}")

    print("\n所有数据集处理完成！")

"""

python blip2_test005.py \
    --model_path /root/autodl-tmp/project/VLM-DETECT-main/weights/ldmFineTune \
    --dataset /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_LDM.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_ADM.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_DDPM.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_IDDPM.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_PNDM.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_StyleGAN.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_Diff-StyleGAN2.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_Diff-ProjectedGAN.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_ProGAN.csv \
              /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_ProjectedGAN.csv \
    --num_samples 300

"""
"""
    2. ** 可视化功能 **：
    新增plot_single_roc函数：为每个数据集绘制单独的ROC曲线并保存
    新增plot_combined_roc函数：将所有数据集的ROC曲线绘制在同一张图中
    新增generate_performance_table函数：生成包含AUC、准确率、F1
    分数的对比表格
    3. ** 结果组织 **：
    为每个数据集创建单独的结果CSV（result_数据集名称.csv）
    单独的ROC曲线保存为（roc_数据集名称.png）
    汇总ROC曲线保存为combined_roc.png
    性能表格保存为performance_table.png
    4. ** 代码结构优化 **：
    将单个数据集的处理逻辑封装为process_dataset函数，提高复用性
    统一管理所有结果数据，便于后续汇总可视化


在blip2_test_fixed.py 基础上修正,保证它会正确执行 BLIP2 的图像+文本推理流程
测试完整测试集
python blip2_test005.py \
    --model_path /root/autodl-tmp/project/VLM-DETECT-main/weights/ddpmFineTune \
    --dataset /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_DDPM.csv

#随机抽取100张测试
python blip2_test005.py \
    --model_path /root/autodl-tmp/project/VLM-DETECT-main/weights/admFineTune \
    --dataset /root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_ADM.csv \
    --num_samples 100
"""

