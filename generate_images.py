import os
import sys

import huggingface_hub

from sd_forge_neo_api import call_ernie_turbo, call_ernie_image

# 将 HuggingFace 权重缓存路径设置为当前目录下的 hf_cache 文件夹
# 必须在导入 torch 和 diffusers 之前设置
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
# 设置 PyTorch 显存分配策略，缓解内存碎片化导致的 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import shutil
import time
import json
from datetime import datetime
import pandas as pd
import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 你的 Excel 文件名
EXCEL_PATH = "data.xlsx"

# 2. 填写你 Excel 中真实的表头列名
PROMPT_COLUMN = "英文提示词"  # 换成你实际的列名
IMAGE_NAME_COLUMN = "图片名"  # 换成你实际的列名

# 3. 模型列表映射字典
# 字典的键将作为最终生成的“文件夹名称”
# 字典的值为对应模型在 HuggingFace 上的库ID，或者是你本地的模型文件夹路径。
MODELS = {
    # 测试ok的
    # "FLUX.2_klein_4B": "black-forest-labs/FLUX.2-klein-4B",
    # "FLUX.2_klein_9B": "black-forest-labs/FLUX.2-klein-9B",  # BF16，用 sequential CPU offload 控制显存
    # "z_image_turbo": "Tongyi-MAI/Z-Image-Turbo",
    # "OpenKolors_v2_1": "lrzjason/OpenKolors_v2_1"
    # "HiDream-O1-Image": "drbaph/HiDream-O1-Image-Dev-FP8",
    # "ERNIE-Image-turbo-api": "Abiray/ERNIE-Image-Turbo-FP8-NVFP4",
    # "Juggernaut_Z": "RunDiffusion/Juggernaut-Z-Image",
    # "Juggernaut-XI-v11": "RunDiffusion/Juggernaut-XI-v11",
    # "Juggernaut-XI-Lightning": "RunDiffusion/Juggernaut-XI-Lightning",
    # "Realistic_Vision_V6.0_B1_noVAE": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    # "ERNIE-Image-api": "baidu/ERNIE-Image",
    "SD3.5_Medium": "stabilityai/stable-diffusion-3.5-medium",
    "FIBO": "briaai/FIBO",
    "GLM-Image": "zai-org/GLM-Image",
}


# ================= 脚本逻辑 =================

def main():
    # 1. 读取 Excel 文件
    if not os.path.exists(EXCEL_PATH):
        print(f"找不到 Excel 文件: {EXCEL_PATH}，请检查路径。")
        return

    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 检查列名是否存在
    if PROMPT_COLUMN not in df.columns or IMAGE_NAME_COLUMN not in df.columns:
        print(f"错误: Excel 中缺少指定的列：'{PROMPT_COLUMN}' 或 '{IMAGE_NAME_COLUMN}'")
        return

    # 解析日志文件，获取已完成的模型列表
    completed_models = set()
    log_file_path = "generation_time_log.txt"
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "生成完毕" in line:
                    start_idx = line.find("模型 ")
                    end_idx = line.find(" 生成完毕")
                    if start_idx != -1 and end_idx != -1:
                        model_name = line[start_idx + 3:end_idx].strip()
                        completed_models.add(model_name)

    # 2. 依次加载每个模型
    for folder_name, model_id in MODELS.items():
        print(f"\n=============================================")

        if folder_name in completed_models:
            print(f"[{folder_name}] 日志显示已生成完毕，跳过此模型。")
            continue

        print(f"[{folder_name}] 正在加载模型: {model_id}...")

        # 如果存在旧的文件夹，直接删除重建，避免未完成的生成影响评测
        model_out_dir = os.path.join("output", folder_name)
        if os.path.exists(model_out_dir):
            print(f"[{folder_name}] 发现残留的文件夹，正在清理以重新生成...")
            shutil.rmtree(model_out_dir)
        os.makedirs(model_out_dir, exist_ok=True)

        try:
            start_time = time.time()

            # 使用 AutoPipeline 自动识别并加载对应架构
            if "flux" in folder_name.lower():
                from diffusers import Flux2KleinPipeline
                pipeline = Flux2KleinPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    cache_dir="./hf_cache"
                )
            elif "SD3.5_Medium" == folder_name:
                from diffusers import StableDiffusion3Pipeline
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    cache_dir="./hf_cache"
                )
            elif "z_image_turbo" == folder_name or "Juggernaut_Z" == folder_name:
                from diffusers import ZImagePipeline
                pipeline = ZImagePipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    cache_dir="./hf_cache",
                    ignore_patterns=["*.gguf", "*fp16*", "*FP8*"]
                )
            elif "api" in folder_name.lower():
                pipeline = None
            elif "Juggernaut-XI-v11" == folder_name or "Juggernaut-XI-Lightning" == folder_name:
                from diffusers import DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=False,
                    cache_dir="./hf_cache"
                )
            elif "openkolors" in folder_name.lower():
                from diffusers import KolorsPipeline, UNet2DConditionModel
                # OpenKolors 提供的是 finetune 后的 UNet，基础组件借用官方 Kolors
                unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir="./hf_cache"
                )
                pipeline = KolorsPipeline.from_pretrained(
                    "Kwai-Kolors/Kolors-diffusers",
                    unet=unet,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir="./hf_cache"
                )
            elif folder_name == "HiDream-O1-Image":
                from huggingface_hub import snapshot_download
                import sys
                hdo1_path = os.path.join(os.getcwd(), "hdo1")
                if hdo1_path not in sys.path:
                    sys.path.append(hdo1_path)
                from fp8_loader import load_image_model

                print(f"[{folder_name}] Downloading/Locating model {model_id}...")
                model_dir = snapshot_download(model_id, cache_dir="./hf_cache")
                processor, model = load_image_model(model_dir)
                pipeline = (processor, model)
            elif folder_name == "FIBO":
                from diffusers import BriaFiboPipeline, BitsAndBytesConfig, BriaFiboTransformer2DModel
                from diffusers.modular_pipelines import ModularPipelineBlocks
                
                print(f"[{folder_name}] Loading VLM prompt-to-JSON model...")
                vlm_pipe = ModularPipelineBlocks.from_pretrained(
                    "briaai/FIBO-VLM-prompt-to-JSON", 
                    trust_remote_code=True, 
                    cache_dir="./hf_cache"
                )
                vlm_pipe = vlm_pipe.init_pipeline()
                
                print(f"[{folder_name}] Loading FIBO Transformer in 8-bit...")
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                transformer = BriaFiboTransformer2DModel.from_pretrained(
                    model_id,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                    cache_dir="./hf_cache"
                )
                
                print(f"[{folder_name}] Loading FIBO Pipeline...")
                fibo_pipe = BriaFiboPipeline.from_pretrained(
                    model_id,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    cache_dir="./hf_cache"
                )
                pipeline = (vlm_pipe, fibo_pipe)
            elif folder_name == "GLM-Image":
                from diffusers.pipelines.glm_image import GlmImagePipeline
                from diffusers import BitsAndBytesConfig
                
                # GLM-Image 也是大型模型，使用 8-bit 量化加载
                print(f"[{folder_name}] Loading with 8-bit quantization...")
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                pipeline = GlmImagePipeline.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir="./hf_cache"
                )
            else:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=False,
                    cache_dir="./hf_cache"
                )
            # 开启模型 CPU 卸载以节省显存，取代 pipeline.to("cuda")
            # 现在 ERNIE 已经换用 FP8 模型，显存占用大幅下降，可以直接使用常规 offload 提升生图速度
            # FLUX.2_klein_9B（BF16 ~18GB）使用逐层 sequential offload，
            # 每次只将当前计算层移到 GPU，峰值显存最低，适合超出 VRAM 的大模型
            if pipeline:
                if folder_name in ("FLUX.2_klein_9B"):
                    pipeline.enable_sequential_cpu_offload()
                else:
                    pipeline.enable_model_cpu_offload()

                # 针对 VAE 开启进一步显存优化
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()

            # 3. 遍历 Excel 里的每一行数据
            for index, row in tqdm(df.iterrows(), total=len(df), desc=f"生成中 ({folder_name})"):
                prompt = str(row[PROMPT_COLUMN])
                img_name = str(row[IMAGE_NAME_COLUMN])

                if pd.isna(prompt) or prompt.strip() == "" or prompt == "nan":
                    continue

                # 4. 生成图像 (明确使用 prompt=prompt 关键字传参，避免部分新模型架构报错)
                kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": 25,
                    "width": 1024,
                    "height": 1024
                }

                # 针对 FLUX.2 等蒸馏模型（Distilled），根据官方文档强制设置 guidance_scale=1.0 并降低步数
                if "flux" in folder_name.lower():
                    kwargs["guidance_scale"] = 1.0
                    kwargs["num_inference_steps"] = 4
                elif folder_name == "z_image_turbo":
                    kwargs["guidance_scale"] = 0.0
                    kwargs["num_inference_steps"] = 9
                elif "Juggernaut_Z" == folder_name:
                    kwargs["guidance_scale"] = 6.0
                    kwargs["num_inference_steps"] = 35
                elif "Juggernaut-XI-v11" == folder_name:
                    kwargs["guidance_scale"] = 5
                    kwargs["num_inference_steps"] = 35
                elif "Juggernaut-XI-Lightning" == folder_name:
                    kwargs["guidance_scale"] = 1.5
                    kwargs["num_inference_steps"] = 5
                elif "ernie" in folder_name.lower():
                    kwargs["guidance_scale"] = 1.0
                    kwargs["num_inference_steps"] = 8
                    kwargs["use_pe"] = True
                elif "openkolors" in folder_name.lower():
                    kwargs["guidance_scale"] = 5.0
                    kwargs["num_inference_steps"] = 50
                    kwargs["negative_prompt"] = ""
                elif "hidream-i1" in folder_name.lower():
                    # Fast 变体均使用 guidance_scale=0.0
                    # Fast: 16 步；固定分辨率 1024x1024
                    kwargs["guidance_scale"] = 0.0
                    kwargs["num_inference_steps"] = 16
                    kwargs["height"] = 1024
                    kwargs["width"] = 1024
                elif "hidream-o1" in folder_name.lower():
                    # Dev 变体推理参数
                    kwargs["guidance_scale"] = 0.0
                    kwargs["num_inference_steps"] = 28
                    kwargs["height"] = 1024
                    kwargs["width"] = 1024
                    kwargs["shift"] = 1.0
                    kwargs["scheduler_name"] = "flash"
                elif "SD3.5_Medium" == folder_name:
                    kwargs["guidance_scale"] = 4.5
                    kwargs["num_inference_steps"] = 40
                elif folder_name == "FIBO":
                    kwargs["guidance_scale"] = 5
                    kwargs["num_inference_steps"] = 50
                elif folder_name == "GLM-Image":
                    kwargs["guidance_scale"] = 1.5
                    kwargs["num_inference_steps"] = 50
                # 针对其他 Turbo / Lightning / Fast 模型，适当降低默认步数提高速度
                elif "lightning" in folder_name.lower() or "turbo" in folder_name.lower() or "fast" in folder_name.lower():
                    kwargs["num_inference_steps"] = 8

                if folder_name == "HiDream-O1-Image":
                    processor, model = pipeline
                    from models.pipeline import generate_image as hdo1_generate_image
                    image = hdo1_generate_image(
                        model=model,
                        processor=processor,
                        prompt=kwargs["prompt"],
                        height=kwargs["height"],
                        width=kwargs["width"],
                        num_inference_steps=kwargs["num_inference_steps"],
                        guidance_scale=kwargs["guidance_scale"],
                        shift=kwargs.get("shift", 1.0),
                        scheduler_name=kwargs.get("scheduler_name", "flash"),
                        seed=42
                    )
                elif folder_name == "FIBO":
                    vlm_pipe, fibo_pipe = pipeline
                    # 1. Natural Prompt -> JSON
                    vlm_output = vlm_pipe(prompt=kwargs["prompt"])
                    json_prompt = vlm_output.values["json_prompt"]
                    
                    # 2. Get Negative Prompt
                    def get_fibo_neg(existing_json: dict) -> str:
                        style_medium = existing_json.get("style_medium", "").lower()
                        if style_medium in ["photograph", "photography", "photo"]:
                            return """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
                        return ""
                    
                    neg_prompt = get_fibo_neg(json.loads(json_prompt))
                    
                    # 3. Generate Image
                    image = fibo_pipe(
                        prompt=json_prompt,
                        num_inference_steps=kwargs["num_inference_steps"],
                        guidance_scale=kwargs["guidance_scale"],
                        negative_prompt=neg_prompt
                    ).images[0]
                    
                    # 可选：保存生成的 JSON prompt 供参考
                    json_save_path = out_path.replace(".png", "_prompt.json")
                    with open(json_save_path, "w", encoding="utf-8") as f:
                        f.write(json_prompt)
                elif folder_name == "GLM-Image":
                    # GLM-Image 使用指定的 generator 种子
                    generator = torch.Generator(device="cuda").manual_seed(42)
                    image = pipeline(**kwargs, generator=generator).images[0]
                elif folder_name == "ERNIE-Image-turbo":  # sd webui forge neo api
                    image = call_ernie_turbo(kwargs["prompt"])
                elif folder_name == "ERNIE-Image":
                    image = call_ernie_image(kwargs["prompt"])
                else:
                    image = pipeline(**kwargs).images[0]

                # 过滤非法字符，防止保存文件名报错
                safe_name = "".join(x for x in img_name if x.isalnum() or x in "._- ")
                out_path = os.path.join("output", folder_name, f"{safe_name}.png")

                # 5. 保存图像
                image.save(out_path)

            end_time = time.time()
            total_seconds = end_time - start_time
            print(f"\n[{folder_name}] 文件夹内所有任务生成完毕！耗时: {total_seconds:.2f} 秒")

            # 记录时间到日志文件
            with open("generation_time_log.txt", "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] 模型 {folder_name} 生成完毕，总耗时: {total_seconds:.2f} 秒\n")
        except Exception as e:
            print(f"[{folder_name}] 模型加载或运行出错: {e}")
            # 把错误记录到日志，以免后续排查时不知道哪坏了
            with open("generation_time_log.txt", "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] 模型 {folder_name} 发生报错跳过，错误: {e}\n")
        finally:
            # 【关键】无论上面是成功跑完，还是因为 OOM 报错，都会走到这里
            # 必须在这里卸载模型并清空显存，否则 OOM 报错后显存依然被占满，后续模型全都会直接 OOM 闪退！
            if 'pipeline' in locals():
                del pipeline
            if 'model' in locals():
                del model
            if 'processor' in locals():
                del processor
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
