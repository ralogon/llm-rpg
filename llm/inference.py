from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import argparse

import config


def import_model(hf_repo):
  hf_repo = 'ralogon/llm-tolkien-spanish'
  config = PeftConfig.from_pretrained(hf_repo)
  model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
  # Load the Lora model
  model = PeftModel.from_pretrained(model, hf_repo)
  return model, tokenizer

def get_inference(model, tokenizer, prompt):

  inputs = tokenizer(prompt, return_tensors="pt")
  tokens = model.generate(
      **inputs,
      max_new_tokens=100,
      temperature=1,
      eos_token_id=tokenizer.eos_token_id,
      early_stopping=True
  )
  print()
  print(30*'=')
  print()
  print("SALIDA DEL MODELO:", tokenizer.decode(tokens[0]))
  print()
  print(30*'=')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run inference on custom model")
  parser.add_argument("--prompt", type=str, help="Initial phrase for the model")
  args = parser.parse_args()
  model, tokenizer = import_model(config.hf_repo)
  get_inference(model, tokenizer, args.prompt)