import torch
import numpy as np
import os
import pandas as pd
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import peft
import argparse
# 设置代理
#os.environ['HTTP_PROXY'] = 'http://127.0.0.1:26561'
#os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:26561'
"""
python blip2_detect.py  --dataset  ./data/Train_CSV_Balanced/train_LDM_balanced.csv   --epochs 20  --batch_size  32   --save_path  ./SaveFineTune/LDM-train-epochs05

Epoch: 0
Loss: 10.497663497924805
Loss: 10.877991676330566
Loss: 10.705880165100098
Loss: 10.481854438781738


"""

# Set random seed for PyTorch

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for NumPy

np.random.seed(RANDOM_SEED)


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


# Model Creation and Initialisation  blip2-opt-2.7b

processor = AutoProcessor.from_pretrained("./blip2-opt-2.7b", use_fast=True)  #"/root/autodl-tmp/project/VLM-DETECT-main/blip2-opt-2.7b""Salesforce/blip2-opt-2.7b"  "Salesforce/blip2-opt-125m"
model = Blip2ForConditionalGeneration.from_pretrained("./blip2-opt-2.7b", device_map="auto") #, load_in_8bit=True


# Low Rank Adaptation Technique Set
# LoraConfig

# def LoraConfig(r, lora_alpha, lora_dropout, bias, target_modules):
#     pass
#
#
# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "k_proj"]
# )
#
#
# def get_peft_model(model, config):
#     pass
#
#
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
config = peft.LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)


def get_peft_model(model, config):
    model = peft.get_peft_model(model, config)
    return model


model = get_peft_model(model, config)
model.print_trainable_parameters()
# Main Body
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Fine-Tune BLIP-2 for Diffusion-based Generated Images Detection.")
    parser.add_argument('--dataset',
                        default='./data/Train_CSV_Balanced/train_LDM_balanced.csv',
                        type=str,
                        help='Path to the training CSV file')
    parser.add_argument('--epochs', default=20, type=int,   #20
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='训练时的 batch size（默认：32）')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='The learning rate for training (default: 5e-5).')
    parser.add_argument('--save_path', type=str, default='./SaveFineTune/LDM-train-epochs05',
                        help='Path to save trained model.')

    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    data = pd.read_csv(opt.dataset)
    train_dataset = ImageCaptioningDataset(data, processor)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn)   #32
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_fn)
    print(f'Training environnement  with : {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    model.train()

    for epoch in range(opt.epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(opt.save_path)
