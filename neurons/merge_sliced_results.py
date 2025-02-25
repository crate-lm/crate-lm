
import os
import json
from tqdm import tqdm
import argparse

EXPLAINER_MODEL_NAME = "mistral-7b"
SIMULATOR_MODEL_NAME = "llama-2-7b"

# test_response = await client.make_request(prompt="test 123<|endofprompt|>", max_tokens=2)
# print("Response:", test_response["choices"][0]["text"])

valid_activation_records = None
scored_simulation = None

# block_name = "ista_anthropic"
# index_method = "anthropic_ooc"

block_name = "ista"
# index_method = "random"
index_method = "top_and_rand"

parser = argparse.ArgumentParser(description='Evaluation.')
parser.add_argument('--model', type=str, help='model name to evaluate')
parser.add_argument('--layer', type=int, help='layer index to evaluate')

args = parser.parse_args()
MODEL = args.model
layer_idx = args.layer

base_path = "/home/ubuntu/nanogpt4crate/neurons"

all_results = {"explanations": [], "scores": []}
for slice_id in range(16):
    if slice_id == 4 or slice_id == 5:
        continue
    filename = f"{base_path}/{block_name}/{MODEL}/{layer_idx}/results_80000samples_{EXPLAINER_MODEL_NAME}_{SIMULATOR_MODEL_NAME}_{index_method}_slice{slice_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results_json = json.load(f)
            all_results["explanations"].extend(results_json["explanations"])
            all_results["scores"].extend(results_json["scores"])

print(f"Loaded {len(all_results)} results")
# save
filename = f"{base_path}/{block_name}/{MODEL}/{layer_idx}/results_80000samples_{EXPLAINER_MODEL_NAME}_{SIMULATOR_MODEL_NAME}_{index_method}.json"
with open(filename, "w") as f:
    json.dump(all_results, f, indent=4)
print(f"Saved to {filename}")
