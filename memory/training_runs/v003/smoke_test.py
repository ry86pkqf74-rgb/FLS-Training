"""Run one inference with the v17 adapter on the held-out test set and dump the result."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/FLS-Training")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER = "/workspace/v17_lora_output/final_adapter"
TEST = Path("/workspace/v003_multimodal/test.jsonl")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, min_pixels=128*28*28, max_pixels=256*28*28)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# Pick the first vision example from test.jsonl
example = None
for line in TEST.read_text().splitlines():
    row = json.loads(line)
    if row.get("metadata",{}).get("vision"):
        example = row
        break
if example is None:
    example = json.loads(TEST.read_text().splitlines()[0])

prompt_messages = example["messages"][:-1]  # system + user, no assistant
target = example["messages"][-1]["content"]

text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
try:
    image_inputs, video_inputs = process_vision_info(prompt_messages)
except Exception:
    image_inputs, video_inputs = None, None

if image_inputs:
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
else:
    inputs = processor.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=0.0)
gen = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("=== METADATA ===")
print(json.dumps(example["metadata"], indent=2))
print("\n=== TARGET (truncated 600c) ===")
print(target[:600])
print("\n=== MODEL OUTPUT (truncated 1200c) ===")
print(gen[:1200])
