import peft
print(peft.__version__)  # 检查版本
# 尝试导入关键模块，确认是否报错
from peft import get_peft_model, LoraConfig