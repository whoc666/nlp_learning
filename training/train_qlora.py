from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk

def main():
    # 1. 加载处理后的数据
    print("加载处理后的数据...")
    dataset_path = "/content/nlp_learning/processed_data"  # 你的数据路径
    tokenized_dataset = load_from_disk(dataset_path)

    # 2. 加载模型和分词器
    model_name = "facebook/opt-125m"
    print(f"加载模型和分词器：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. 配置训练参数
    training_args = TrainingArguments(
        output_dir="/content/nlp_learning/output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="/content/nlp_learning/logs",
        logging_steps=100,
        report_to=[],  # 禁用 wandb
    )

    # 4. 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 5. 开始训练
    print("开始训练...")
    trainer.train()
    print("训练完成！")

if __name__ == "__main__":
    main()
