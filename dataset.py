from torch.utils.data import Dataset
import PIL.Image as Image

from PIL import ImageFile
# 允许加载截断的图片（防止 OSError: image file is truncated）
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCaptioningDataset(Dataset):
    """
    Image Captioning Dataset Class
    支持损坏图片自动跳过，增强容错能力
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset.loc[idx]
        image_path = item["image"]
        try:
            # 尝试打开并完整加载图片
            image = Image.open(image_path)
            image.load()
        except (Image.UnidentifiedImageError, OSError) as e:
            print(f"[警告] 跳过损坏或无法识别的图片：{image_path}，错误类型：{type(e).__name__}")
            # 若当前图片无效，则递归尝试下一个样本
            next_idx = (idx + 1) % len(self.dataset)
            return self.__getitem__(next_idx)
        # 使用 BLIP2 处理器编码图像
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        # 去掉 batch 维度
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]

        return encoding







# from torch.utils.data import Dataset
# import PIL.Image as Image
#
#
# # Image Captioning Dataset Class
#
# class ImageCaptioningDataset(Dataset):
#     def __init__(self, dataset, processor):
#         self.dataset = dataset
#         self.processor = processor
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         item = self.dataset.loc[idx]
#         try:
#             # 尝试打开图片
#             image = Image.open(item["image"])
#         except Image.UnidentifiedImageError:
#             print(f"跳过损坏/无法识别的图片：{item['image']}")
#             # 可返回一个默认样本或跳过（需结合数据集逻辑处理）
#             return self.__getitem__((idx + 1) % len(self.dataset))  # 示例：跳过当前样本，取下一个
#
#         encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
#         # remove batch dimension
#         encoding = {k: v.squeeze() for k, v in encoding.items()}
#         encoding["text"] = item["text"]
#         return encoding