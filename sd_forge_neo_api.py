import base64
import io

import requests
from PIL import Image


def call_ernie_turbo(prompt: str):
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    payload =  {
        "prompt": prompt,
        "steps": 8,
        "cfg_scale": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler_name": "Euler",  # 注意：不是 Euler a
        "scheduler": "Simple",   # 注意：必须是 Simple
        "override_settings": {
            "sd_model_checkpoint": "ernie-image-turbo-fp8.safetensors",
            "forge_additional_modules": [
                "flux2-vae.safetensors",
                "ministral-3-3b.safetensors"
            ],
            "forge_preset": "ernie",
            "ernie_t2i_dcfg": 3.0  # 手动指定 Shift
        }
    }

    response = requests.post(url, json=payload)
    r = response.json()
    return Image.open(io.BytesIO(base64.b64decode(r['images'][0])))

def call_ernie_image(prompt: str):
    url = "http://127.0.0.1:7861/sdapi/v1/txt2img"
    payload =  {
        "prompt": prompt,
        "steps": 50,
        "cfg_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "sampler_name": "Euler",  # 注意：不是 Euler a
        "scheduler": "Simple",   # 注意：必须是 Simple
        "override_settings": {
            "sd_model_checkpoint": "ernie-image-fp8.safetensors",
            "forge_additional_modules": [
                "flux2-vae.safetensors",
                "ministral-3-3b.safetensors"
            ],
            "forge_preset": "ernie",
            "ernie_t2i_dcfg": 3.0  # 手动指定 Shift
        }
    }

    response = requests.post(url, json=payload)
    r = response.json()
    return Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
