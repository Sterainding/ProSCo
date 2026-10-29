import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_model(model_name_or_path: str, local_dir: str = "./hf_models"):
    """
    检查 Hugging Face 模型是否已下载，没有就自动下载
    model_name_or_path: 模型名（如 "t5-base"）或本地路径
    local_dir: 本地存储模型的目录
    """
    local_dir = Path(local_dir)
    model_path = local_dir / model_name_or_path.replace("/", "_")

    # t5-base模型绝对路径
    model_path = model_path.resolve()

    # 如果已经有本地目录，直接用
    if model_path.exists():
        print(f"[INFO] Found local model: {model_path}")
        return str(model_path)

    # 如果传的是已有路径
    if Path(model_name_or_path).exists():
        print(f"[INFO] Using provided local model path: {model_name_or_path}")
        return model_name_or_path

    # 没有本地缓存，开始下载
    print(f"[INFO] Downloading model '{model_name_or_path}' from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=model_path)
    print(f"[INFO] Model downloaded to {model_path}")
    return str(model_path)


if __name__ == "__main__":
    model_path = get_model("t5-base")
    print(f"✅ Model ready at: {model_path}")
