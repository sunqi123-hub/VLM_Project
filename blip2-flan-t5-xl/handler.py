from typing import Dict, List, Any
# import transformers
# from transformers import AutoTokenizer
# import torch
from datetime import datetime

import torch

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class EndpointHandler():
    
    def __init__(self, path=""):
        
        self.processor = Blip2Processor.from_pretrained(path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16 , device_map="auto")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        logging.info('Model moved to device-' + self.device)
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.eval()
        # self.model.to(device=device, dtype=self.torch_dtype)
        
        # self.generate_kwargs = {
        #     'max_new_tokens': 512,
        #     'temperature': 0.0001,
        #     'top_p': 1.0,
        #     'top_k': 0,
        #     'use_cache': True,
        #     'do_sample': True,
        #     'eos_token_id': self.tokenizer.eos_token_id,
        #     'pad_token_id': self.tokenizer.pad_token_id,
        #     "repetition_penalty": 1.1
        # }
    
    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """

        # streamer = TextIteratorStreamer(
        # self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        # )
        
        ## Model Parameters
        # self.generate_kwargs['max_new_tokens'] = data['max_new_tokens'] if 'max_new_tokens' in data else self.generate_kwargs['max_new_tokens']
        # self.generate_kwargs['temperature'] = data['temperature'] if 'temperature' in data else self.generate_kwargs['temperature']
        # self.generate_kwargs['top_p'] = data['top_p'] if 'top_p' in data else self.generate_kwargs['top_p']
        # self.generate_kwargs['top_k'] = data['top_k'] if 'top_k' in data else self.generate_kwargs['top_k']
        # self.generate_kwargs['do_sample'] = data['do_sample'] if 'do_sample' in data else self.generate_kwargs['do_sample']
        # self.generate_kwargs['repetition_penalty'] = data['repetition_penalty'] if 'repetition_penalty' in data else self.generate_kwargs['repetition_penalty']
        
        
        ## Prepare the inputs
        batch_size = data.pop("batch_size",data)
        # input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids
        # input_ids = input_ids.to(self.model.device)


        # pip install accelerate
        
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 

        now = datetime.now()

        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        
        # question = "how many dogs are in the picture?"
        # inputs = self.processor(raw_image, question, return_tensors="pt").to("cuda")

        inputs = self.processor([raw_image]*batch_size, return_tensors="pt").to("cuda", torch.float16)
        
        out = self.model.generate(**inputs)

        # generated_text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        generated_text = self.processor.batch_decode(out, skip_special_tokens=True)

        current = datetime.now()

        # encoded_inp = self.tokenizer(inputs, return_tensors='pt', padding=True)
        # for key, value in encoded_inp.items():
        #     encoded_inp[key] = value.to('cuda:0')

        ## Invoke the model     
        # with torch.no_grad():
        #     gen_tokens =  self.model.generate(
        #         input_ids=encoded_inp['input_ids'],
        #         attention_mask=encoded_inp['attention_mask'],
        #         **generate_kwargs,
        #     )

        # ## Decode using tokenizer
        # decoded_gen = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)        

        # with torch.no_grad():
        #     output_ids = self.model.generate(input_ids, **self.generate_kwargs)
        # # Slice the output_ids tensor to get only new tokens
        # new_tokens = output_ids[0, len(input_ids[0]) :]
        # output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return [{"gen_text":generated_text, "time_elapsed": str(current-now)}]
