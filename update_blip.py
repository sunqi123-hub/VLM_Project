# Load your model and processor and run the following to update BLIP-2 model
# It will update file in your repo by adding new args in configs and resizing embedding layer
# Then you'll be able to run BLIP-2 without warnings/errors
from platform import processor
from typing import Any

from transformers import AddedToken
import peft

config = peft.LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)


def get_peft_model(model: object, config: object) -> Any:
    model = peft.get_peft_model(model, config)
    return model

model = get_peft_model(config)
processor.num_query_tokens = model.config.num_query_tokens
image_token = AddedToken("<image>", normalized=False, special=True)
processor.tokenizer.add_tokens([image_token], special_tokens=True)

model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
model.config.image_token_index = len(processor.tokenizer) - 1

model.push_to_hub("/sdata/sunqi/pycharm_test/VLM-DETECT-main/blip2-opt-2.7b/")
processor.push_to_hub("/sdata/sunqi/pycharm_test/VLM-DETECT-main/blip2-opt-2.7b/")
