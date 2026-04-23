import pandas as pd
from sklearn.utils import resample
import os

# 读取原始训练集
df = pd.read_csv("./data/Train_CSV/train_StyleGAN.csv")   #改

# 清理标签
df["text"] = df["text"].astype(str).str.lower().str.strip()

# 分出 fake 和 real
fake_df = df[df["text"] == "fake"]
real_df = df[df["text"] == "real"]

print("原始比例：")
print(df["text"].value_counts())

# 下采样 fake，使 fake:real = 1:1
fake_down = resample(fake_df, replace=False, n_samples=len(real_df), random_state=42)

# 合并 & 打乱
balanced_df = pd.concat([fake_down, real_df]).sample(frac=1, random_state=42)

# 保存
os.makedirs("./data/Train_CSV_Balanced", exist_ok=True)
balanced_df.to_csv("./data/Train_CSV_Balanced/train_StyleGAN_balanced.csv", index=False)    #改

print("\n✅ 平衡后数据集已保存到: ./data/Train_CSV_Balanced/train_StyleGAN_balanced.csv")  #改
print(balanced_df["text"].value_counts())

# ADM  DDPM  Diff-ProjectedGAN   Diff-StyleGAN2  IDDPM   LDM  PNDM  ProGAN  ProjectedGAN  StyleGAN
#  python Train_CSV_Balanced.py