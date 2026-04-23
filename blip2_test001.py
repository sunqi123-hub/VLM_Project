# python blip2_test001.py     --model_path /root/autodl-tmp/project/VLM-DETECT-main/SaveFineTune   --dataset /root/autodl-tmp/project/VLM-DETECT-main/data/test-adm.csv

import os
import PIL
import pandas as pd
import numpy as np
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
import tqdm
import time
import argparse

# Set random seed for PyTorch
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for NumPy
np.random.seed(RANDOM_SEED)


# Custom Round Function
def multiple_custom_round(values):
    result = []
    for value in values:
        if value > 0.6:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)


# Map Text to Binary Value
# def map_text_to_binary(text):
#     if text == "fake":
#         return 1
#     elif text == "real":
#         return 0
#     else:
#         return None
def map_text_to_binary(text):
    text_lower = text.lower().strip()  # 去除空格并小写
    if "fake" in text_lower:
        return 1
    elif "real" in text_lower:
        return 0
    else:
        print(f"警告：无法识别的文本 '{text}'，已默认标记为0")
        return 0


def collate_fn(batch):
    # pad the input_ids and attention_mask
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test Fine-Tuned BLIP-2 for Diffusion-based Generated Images Detection.")
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/project/VLM-DETECT-main/SaveFineTune/LDM',
                        help='Path to the trained model.')
    parser.add_argument('--dataset', default='/root/autodl-tmp/project/VLM-DETECT-main/data/Test_CSV/test_LDM.csv', type=str,
                        help='Path to the testing CSV file')

    opt = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PeftConfig.from_pretrained(opt.model_path)
    local_model_path = "/root/autodl-tmp/project/VLM-DETECT-main/blip2-opt-2.7b"  # 修改为本地模型路径
    model = Blip2ForConditionalGeneration.from_pretrained(local_model_path, load_in_8bit=True, device_map="auto")
    model = PeftModel.from_pretrained(model, opt.model_path)
    processor = AutoProcessor.from_pretrained(local_model_path, use_fast=True)

    # 读取完整测试集
    test_df = pd.read_csv(opt.dataset)


    # 提取前100张和后100张样本（确保数据集总行数 >= 200，否则tail(100)会取实际剩余数量）
    # 前100张
    test_df_head = test_df.head(300)
    # 后100张
    test_df_tail = test_df.tail(700)
    # 合并为新的测试集（共200张），并重置索引
    test_df = pd.concat([test_df_head, test_df_tail], ignore_index=True)

    # 只测试少量数据（取前10个样本，可修改N的值）
    # N = 11000  # 自定义测试样本数量
    # test_df = test_df.head(N)  # 截取前N行数据

    # 随机采样N个样本（可复现）
    # N = 11000
    # test_df = test_df.sample(n=N, random_state=RANDOM_SEED)  # random_state保证结果可复现

    test_dataset = ImageCaptioningDataset(test_df, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    result = []
    start_time = time.time()

    for batch in tqdm.tqdm(test_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        # 在生成前添加提示词
        prompt = "Is this image fake or real? Answer with 'fake' or 'real'."
        prompt_inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = prompt_inputs["input_ids"]

        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,  # 传入提示词
            max_new_tokens=10,
            num_beams=5,
            early_stopping=True
        )

        result.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    # result_df = pd.DataFrame({
    #     'image': test_df['image'],
    #     'GT': test_df['text'],  # GT : GroundTruth
    #     'Tlabel': test_df['text'].apply(map_text_to_binary),  # 转换为二进制标签：0/1
    #     'Pred': result,
    #     'Plabel': [map_text_to_binary(x) for x in result]
    # })
    # 生成result_df后处理None值
    result_df = pd.DataFrame({
        'image': test_df['image'],
        'GT': test_df['text'],
        'Tlabel': test_df['text'].apply(map_text_to_binary),
        'Pred': result,
        'Plabel': [map_text_to_binary(x) for x in result]
    })
    # 确保无None值（双重保险）
    result_df['Plabel'] = result_df['Plabel'].fillna(0)
    result_df['Tlabel'] = result_df['Tlabel'].fillna(0)  # 理论上Tlabel不应有None，但可加保险

    fpr, tpr, th = metrics.roc_curve(result_df['Tlabel'], result_df['Plabel'])
    auc = metrics.auc(fpr, tpr)
    preds = multiple_custom_round(np.asarray(result_df['Plabel']))
    accuracy = accuracy_score(result_df['Plabel'], result_df['Tlabel'])
    f1Score = f1_score(torch.tensor(result_df['Tlabel']), torch.tensor(result_df['Plabel']))

    print("METRICS")
    print("AUC: ", auc, "| Accuracy: ", accuracy, "| F1-Score: ", f1Score)
    print('', str(round(auc * 100, 2)), "|", str(round(accuracy * 100, 2)), "|", str(round(f1Score * 100, 2)))

