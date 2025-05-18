# CLI 推理脚本
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main(model_dir, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成文本：", text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output", help="模型路径")
    parser.add_argument("--prompt", type=str, default="Hello, NLP learning assistant!", help="输入提示语")
    args = parser.parse_args()
    main(args.model_dir, args.prompt)
