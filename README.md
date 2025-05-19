
# NLP 学习助手（NLP Learning Assistant）

这是一个基于 Hugging Face 平台构建的 NLP 学习助手项目，涵盖了数据预处理、模型微调、在线部署（Spaces）等完整流程。适合 NLP 初学者理解和体验大模型训练及部署的流程。

---

## 📦 项目结构

- 🤗 **数据集**：`whoc666/nlp_learing_dataset`
  - 包含维基百科与 Arxiv 的英文摘要数据，格式为 JSONL，字段为 `{"text": ...}`。
  - 链接：[Hugging Face Dataset](https://huggingface.co/datasets/whoc666/nlp_learning_dataset)

- 🤗 **模型仓库**：`whoc666/nlp_learning_model`
  - 使用 `facebook/opt-125m` 模型微调得到，支持基础的语言建模能力。
  - 链接：[Hugging Face Model](https://huggingface.co/whoc666/nlp_learning_model)

---

## 🧠 模型信息

- 基础模型：`facebook/opt-125m`
- 使用 `Trainer` API 微调，支持 Causal LM 任务
- 数据集经过 `AutoTokenizer` 分词，标签为输入的 input_ids 复制

---

## 🛠️ 使用方法

你可以下载本项目并在本地运行或修改：

```bash
git clone https://huggingface.co/spaces/whoc666/nlp_learning_model
cd nlp_learning_model
```

---

## 📜 License

本项目默认使用 `apache-2.0` 开源协议，你可以根据自己的需要进行修改。

---

## 🙌 致谢

感谢以下开源工具和社区的支持：

- Hugging Face Transformers
- Hugging Face Datasets
- Gradio
- Google Colab

