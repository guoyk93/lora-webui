import re

import torch
from safetensors.torch import save_file


def convert_lora_file_to_safetensors(src: str, dst: str):
    if torch.cuda.is_available():
        checkpoint = torch.load(src, map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(src, map_location=torch.device('cpu'))

    new_dict = dict()
    for idx, key in enumerate(checkpoint):
        new_key = re.sub(r'\.processor\.', '_', key)
        new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
        new_key = re.sub('_lora.up.', '.lora_up.', new_key)
        new_key = re.sub('_lora.down.', '.lora_down.', new_key)
        new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
        new_key = re.sub('to_out', 'to_out_0', new_key)
        new_key = 'lora_unet_' + new_key
        new_dict[new_key] = checkpoint[key]

    save_file(new_dict, dst)
