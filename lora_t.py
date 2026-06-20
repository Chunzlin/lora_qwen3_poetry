"""
LoRA 微调：加载预处理数据集，训练并验证生成。
使用方式：python train.py --config config.yaml
"""

import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # 固定随机种子
    set_seed(config["seed"])

    # 检测设备
    use_gpu = torch.cuda.is_available()
    print(f"GPU 可用: {use_gpu}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（根据设备选择量化）
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_dir"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        fp16 = True
        print("模型加载至 GPU (4bit)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_dir"],
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        fp16 = False
        print("模型加载至 CPU (fp32)，请确保内存充足")

    model.config.use_cache = False

    # 加载预处理数据集
    dataset = load_from_disk(config["processed_data_path"])
    print(f"数据集样本数: {len(dataset)}")

    # LoRA 配置
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        fp16=fp16,
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        remove_unused_columns=False,
        # report_to="tensorboard",
        logging_dir="./logs",
        seed=config["seed"],
        dataloader_pin_memory=False if not use_gpu else True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n开始训练...")
    trainer.train()
    print("训练完成")

    # 保存 LoRA adapter
    model.save_pretrained(config["adapter_dir"])
    tokenizer.save_pretrained(config["adapter_dir"])
    print(f"LoRA adapter 已保存至 {config['adapter_dir']}")

    # 绘制并保存 loss 曲线
    log_history = trainer.state.log_history
    losses = [e["loss"] for e in log_history if "loss" in e]
    steps = [e["step"] for e in log_history if "loss" in e]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("Loss 曲线已保存至 training_loss.png")

    # 写入文本日志
    with open("training_log.txt", "w") as f:
        for s, l in zip(steps, losses):
            f.write(f"Step {s}: loss = {l:.6f}\n")
    print("训练日志已保存至 training_log.txt")

    # ===== 验证生成 =====
    del model
    if use_gpu:
        torch.cuda.empty_cache()

    # 重新加载基础模型和 adapter 用于推理
    if use_gpu:
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
    print("加载训练好的 LoRA 模型用于验证")

    generation_config = {
        "max_new_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    results = []
    for bg in config["test_backgrounds"]:
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": bg}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):].strip()
        results.append(f"背景：{bg}\n生成诗词：\n{answer}\n")
        print(f"\n背景：{bg}\n生成诗词：\n{answer}")

    with open("generated_poems.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("验证结果已保存至 generated_poems.txt")

if __name__ == "__main__":
    main()