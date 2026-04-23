import os
import pandas as pd
#ADM  DDPM  Diff-ProjectedGAN   Diff-StyleGAN2  IDDPM   LDM  PNDM  ProGAN  ProjectedGAN  StyleGAN  Real
#  python train1.py
# 定义真实图像和合成图像的多个目录
real_image_dirs = ['/root/autodl-tmp/project/VLM-DETECT-main/data/test/Real/']
#fake_image_dirs = ['/root/autodl-tmp/project/VLM-DETECT-main/data/train/ADM/1_fake/']
fake_image_dirs = ['/root/autodl-tmp/project/VLM-DETECT-main/data/train/StyleGAN/1_fake/']
# 创建数据列表
data = []
# 遍历真实图像目录（real 对应 label=0）
for real_image_dir in real_image_dirs:
    for filename in os.listdir(real_image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(real_image_dir, filename)
            data.append({'image': image_path, 'text': 'real', 'label': 0})  # 新增 label 列
# 遍历合成图像目录（fake 对应 label=1）
for fake_image_dir in fake_image_dirs:
    for filename in os.listdir(fake_image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(fake_image_dir, filename)
            data.append({'image': image_path, 'text': 'fake', 'label': 1})  # 新增 label 列

# 将数据列表转换为 DataFrame
df = pd.DataFrame(data)
# 保存 DataFrame 为 CSV 文件
df.to_csv('/root/autodl-tmp/project/VLM-DETECT-main/data/Train_CSV/train_StyleGAN.csv', index=False)