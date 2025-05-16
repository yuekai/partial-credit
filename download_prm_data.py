from datasets import load_dataset
import json
from tqdm import tqdm

def process_row(row):
  rows = []
  trace = ""
  for msg in row["conversations"]:
    if msg["role"] == "user":
      trace += f"{msg['content']}\n"
      continue
    elif msg["role"] == "assistant":
      rows.append({"messages": [{"role": "user", "content": trace.rstrip("\n")}, {"role": "assistant", "content": msg['content']}]})
    else:
      raise ValueError(f"Unknown role: {msg['role']}")
  return rows

def save_jsonl(data, file_path):
  with open(file_path, 'w') as f:
    for item in data:
      json.dump(item, f)
      f.write(f"\n")

if __name__ == "__main__":
  ds = load_dataset("RLHFlow/Mistral-PRM-Data", cache_dir="data")
  data = []
  for row in tqdm(ds["train"]):
    data.extend(process_row(row))
  save_jsonl(data, "data/mistral_pcm_data.jsonl")