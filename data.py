"""
数据预处理：将对话 JSONL 转换为 Qwen ChatML 格式并 tokenize，保存为 HuggingFace Dataset。
使用方式：python prepare_data.py --config config.yaml
"""

import argparse
import yaml
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_instruction(messages, tokenizer):
    """返回 prompt 和完整文本"""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt, full

def tokenize_function(examples, tokenizer, max_length):
    all_input_ids, all_labels = [], []
    for msgs in examples["messages"]:
        prompt_str, full_str = format_instruction(msgs, tokenizer)
        # 计算 prompt 长度（token 级别）
        prompt_ids = tokenizer(prompt_str, truncation=False)["input_ids"]
        prompt_len = len(prompt_ids)
        # 完整文本 tokenization
        tokenized = tokenizer(full_str, truncation=True, max_length=max_length, padding=False)
        input_ids = tokenized["input_ids"]
        # 构建 labels，prompt 部分设为 -100
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "labels": all_labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # 加载 tokenizer
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 读取原始 JSONL（每行一个 {"messages": [...]}）
    print(f"正在读取原始数据: {config['data_path']}...")
    max_data = 100  # 可选：限制处理数据量，调试时使用
    with open(config["data_path"], "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_data]
        raw_data = [json.loads(line) for line in lines]

    print(f"读取 {len(raw_data)} 条数据")

    # 转换为 Dataset
    dataset = Dataset.from_list(raw_data)

    # 显示第一条以验证格式
    print("\n=== 第一条格式化输入验证 ===")
    sample = raw_data[0]["messages"]
    prompt, full = format_instruction(sample, tokenizer)

    print("完整文本:")
    print(full)
    print("\nPrompt:")
    print(prompt)
    tokenized_sample = tokenize_function({"messages": [sample]}, tokenizer, config["max_length"])
    decoded = tokenizer.decode(tokenized_sample["input_ids"][0], skip_special_tokens=False)
    print("\n解码后:")
    print(decoded)
    print("=" * 50 + "\n")

    # 对整个数据集 tokenize
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 保存处理后的数据集
    dataset.save_to_disk(config["processed_data_path"])
    print(f"预处理数据集已保存至 {config['processed_data_path']}")

if __name__ == "__main__":
    main()