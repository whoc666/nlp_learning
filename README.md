# NLP LLaMA3 Project

本项目旨在构建一个基于大语言模型（LLM）的 NLP 学习助手。

## 📁 项目结构

```
nlp_llama3_project/
├── data/                  # 数据集原始数据与清洗代码
│   ├── raw/               # 原始数据文件
│   ├── processed/         # 清洗后的数据
│   └── prepare_data.py    # 数据清洗脚本
├── training/              # 模型训练相关代码
│   ├── train_qlora.py     # 微调主程序
│   └── training_utils.py  # LoRA配置与训练辅助
├── inference/             # 推理测试代码
│   ├── infer.py           # CLI 推理
│   └── tokenizer_loader.py# 分词器加载模块
├── gradio_app/            # Gradio 可视化界面与部署
│   ├── app.py             # Gradio 主程序
│   └── requirements.txt   # HF Spaces 所需依赖
├── scripts/               # 工具脚本
│   └── hf_upload_model.py # 上传模型到 Hugging Face
├── README.md              # 项目说明文件
├── LICENSE                # 开源协议（MIT）
└── .gitignore             # 忽略项配置
```

## 🚀 功能

- 数据清洗与预处理
- 轻量化 LLM 微调（QLoRA）
- CLI 推理接口
- Gradio 前端部署
- Hugging Face Hub 自动上传

## ✅ 使用指南

1. 准备数据并运行 `data/prepare_data.py`
2. 使用 `training/train_qlora.py` 微调模型
3. 运行 `inference/infer.py` 进行命令行测试
4. 启动 `gradio_app/app.py` 查看界面交互
5. 上传模型 `scripts/hf_upload_model.py`

---

本项目支持免费 GPU 环境（如 Colab），适合教育学习用途。
