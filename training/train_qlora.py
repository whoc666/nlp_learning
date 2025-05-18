from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torch

def train():
    dataset = load_from_disk("/content/processed_data")
    print(f"加载数据集，样本数：{len(dataset)}")

    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )
    model.config.use_cache = False  # 避免 gradient checkpointing 报错

    print("模型和分词器加载完成")

    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA模型准备完成")

    training_args = TrainingArguments(
        output_dir="./lora_tinyllama",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,  # 先跑一轮试试
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="no",
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("开始训练...")
    trainer.train()

    trainer.save_model("./lora_tinyllama")
    print("模型保存完成，路径：./lora_tinyllama")

if __name__ == "__main__":
    train()
