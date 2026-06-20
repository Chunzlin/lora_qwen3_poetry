"""
交互式诗词生成：用户输入创作背景，LoRA 模型实时生成诗词。
用法：
    python interactive.py --config config.yaml
    输入 'quit' 或 'exit' 退出
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config):
    """根据设备加载基础模型、分词器和 LoRA adapter"""
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_dir"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_dir"],
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    model = PeftModel.from_pretrained(base_model, config["adapter_dir"])
    print(f"已加载 LoRA 适配器: {config['adapter_dir']}")
    return model, tokenizer

def generate_poem(model, tokenizer, background, config):
    """根据背景生成一首诗词"""
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": background}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_config = {
        "max_new_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = generated[len(prompt):].strip()
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(config)

    print("\n" + "=" * 50)
    print("交互式诗词创作已启动")
    print("你可以直接输入创作背景（如：深秋夜晚，思念故乡），按回车生成诗词")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 50 + "\n")

    while True:
        try:
            background = input("请输入创作背景：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if background.lower() in ["quit", "exit", "q"]:
            print("退出诗词创作。再见！")
            break

        if not background:
            print("背景不能为空，请重新输入。\n")
            continue

        poem = generate_poem(model, tokenizer, background, config)
        print("\n生成诗词：")
        print(poem)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()