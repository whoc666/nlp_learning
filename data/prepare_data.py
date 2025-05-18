import os
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_tokenize(data_url: str, local_raw_path: str, processed_path: str, tokenizer_name="bert-base-uncased"):
    """
    加载jsonl数据，保存本地，转成datasets.Dataset，做tokenize，保存处理后的数据集到磁盘
    
    Args:
        data_url: str，远程数据文件URL（如Hugging Face上的jsonl链接）
        local_raw_path: str，本地保存原始jsonl文件路径
        processed_path: str，本地保存处理后数据集路径
        tokenizer_name: str，使用的预训练分词器名称
    
    Returns:
        datasets.Dataset，处理好的tokenized数据集
    """

    # 1. 下载数据文件到本地（如果本地不存在）
    if not os.path.exists(local_raw_path):
        import requests
        print(f"开始下载数据到 {local_raw_path} ...")
        r = requests.get(data_url)
        with open(local_raw_path, "wb") as f:
            f.write(r.content)
        print("下载完成！")
    else:
        print(f"本地已有数据文件：{local_raw_path}")

    # 2. 读取jsonl文件，将每行json加载为list
    data_list = []
    with open(local_raw_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"读取到 {len(data_list)} 条数据")

    # 3. 转换成datasets.Dataset格式
    dataset = Dataset.from_list(data_list)

    # 4. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 5. 定义tokenize函数，截断padding到max_length=512
    def tokenize_function(examples):
        # 注意这里假设每条样本有'text'字段
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # 6. 对整个数据集做tokenize
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. 保存处理后的数据集到本地
    tokenized_dataset.save_to_disk(processed_path)
    print(f"处理后数据集已保存到 {processed_path}")

    return tokenized_dataset


if __name__ == "__main__":
    # 测试代码
    DATA_URL = "https://huggingface.co/datasets/whoc666/nlp_learing/resolve/main/1_data_en_wiki_arxiv.jsonl"
    RAW_PATH = "./raw/1_data_en_wiki_arxiv.jsonl"
    PROCESSED_PATH = "./processed/processed_dataset"
    
    os.makedirs("./raw", exist_ok=True)
    os.makedirs("./processed", exist_ok=True)
    
    dataset = load_and_tokenize(DATA_URL, RAW_PATH, PROCESSED_PATH)
    print(dataset)
