# Gradio 可视化界面

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "../output"  # 模型路径

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=generate_text, inputs="text", outputs="text",
                     title="NLP Learning Assistant",
                     description="基于微调模型的NLP学习助手")

if __name__ == "__main__":
    iface.launch()
