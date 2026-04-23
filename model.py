import os
# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:26561'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:26561'
from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("/sdata/sunqi/pycharm_test/VLM-DETECT-main/blip2-opt-2.7b", device_map="auto", load_in_8bit=True)

# 保存模型和处理器
# processor.save_pretrained("/sdata/sunqi/pycharm_test/VLM-DETECT-main/local_model/")
# model.save_pretrained("/sdata/sunqi/pycharm_test/VLM-DETECT-main/local_model/")
processor.save_pretrained("/sdata/sunqi/pycharm_test/VLM-DETECT-main/SaveFineTune/")
model.save_pretrained("/sdata/sunqi/pycharm_test/VLM-DETECT-main/SaveFineTune/")