import json
import urllib.request
from datasets import Dataset
from transformers import AutoTokenizer

def download_data(url: str, local_path: str):
    """
    下载数据文件到本地

    参数：
    - url: 数据文件的下载链接
    - local_path: 保存到本地的文件路径
    """
    print(f"开始下载数据：{url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"数据下载完成，保存路径：{local_path}")

def prepare_data(jsonl_path: str, tokenizer_name: str = "bert-base-uncased", save_path: str = "./processed_data"):
    """
    加载 jsonl 格式数据，转成 Dataset，进行分词处理并保存

    参数：
    - jsonl_path: jsonl 数据文件路径
    - tokenizer_name: 预训练分词器名称，默认bert-base-uncased
    - save_path: 处理后数据保存路径
    """
    # 1. 读取jsonl数据文件
    data_list = []
    print(f"开始读取数据文件：{jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    print(f"共读取 {len(data_list)} 条数据")

    # 2. 创建 Dataset 对象
    dataset = Dataset.from_list(data_list)
    print("Dataset 创建成功，示例数据：")
    print(dataset[0])

    # 3. 加载分词器
    print(f"加载分词器：{tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 4. 定义分词函数，添加labels字段（用于监督学习）
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # 目标标签为输入id的复制
        return tokenized

    # 5. 批量map分词，去除原始text列
    print("开始进行分词处理...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("分词完成，示例tokenized数据：")
    print(tokenized_dataset[0])

    # 6. 保存处理后的数据
    tokenized_dataset.save_to_disk(save_path)
    print(f"处理后的数据已保存到 {save_path}")

if __name__ == "__main__":
    # 测试运行时的默认参数，可以根据实际路径和需要调整
    data_url = "https://huggingface.co/datasets/whoc666/nlp_learing/resolve/main/1_data_en_wiki_arxiv.jsonl"
    local_file_path = "/content/1_data_en_wiki_arxiv.jsonl"
    download_data(data_url, local_file_path)
    prepare_data(local_file_path)
