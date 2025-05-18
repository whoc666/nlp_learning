from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import Trainer, TrainingArguments

def train():
    # 1. 加载预处理后的数据集
    dataset = load_from_disk("/content/processed_data")

    # 2. 加载模型和分词器
    model_name = "TheBloke/tinyLlama-7B"  # 你的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

    # 3. 准备模型进行 LoRA 微调
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

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./lora_tinyllama",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
        push_to_hub=False,
    )

    # 5. 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 6. 开始训练
    trainer.train()

    # 7. 保存模型
    trainer.save_model("./lora_tinyllama")

if __name__ == "__main__":
    train()
