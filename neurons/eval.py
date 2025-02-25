
import os
import json
from tqdm import tqdm
import argparse

os.environ["OPENAI_API_KEY"] = ""

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator

# EXPLAINER_MODEL_NAME = "gpt-4"
# EXPLAINER_MODEL_NAME = "gpt-3.5-turbo"
EXPLAINER_MODEL_NAME = "mistral-7b"
# EXPLAINER_MODEL_NAME = "zeroablate"
# SIMULATOR_MODEL_NAME = "babbage-002"
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
parser.add_argument('model', type=str, help='model name to evaluate')
parser.add_argument('layer', type=int, help='layer index to evaluate')

args = parser.parse_args()
MODEL = args.model
layer_idx = args.layer

print(f"""MODEL={MODEL}, layer_idx={layer_idx}""")
async def run():
    global valid_activation_records
    global scored_simulation
    
    # hot restart
    base_path = "/home/ubuntu/nanogpt4crate/neurons"
    filename = f"{base_path}/{block_name}/{MODEL}/{layer_idx}/results_80000samples_{EXPLAINER_MODEL_NAME}_{SIMULATOR_MODEL_NAME}_{index_method}.json"
        
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results_json = json.load(f)
    else:
        results_json = {"explanations": [], "scores": []}
    already_done = len(results_json["explanations"])

    # neuron_idx = 3
    for neuron_idx in tqdm(range(already_done, 512)):
        # Load a neuron record.
        dataset_path = f"{base_path}/{block_name}/{MODEL}/"
        neuron_record = load_neuron(layer_idx, neuron_idx, dataset_path)

        # Grab the activation records we'll need.
        train_activation_records = neuron_record.train_activation_records()
        valid_activation_records = neuron_record.valid_activation_records()

        # Generate an explanation for the neuron.
        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
        )
        assert len(explanations) == 1
        explanation = explanations[0]
        # print(f"{explanation=}")
        results_json["explanations"].append(explanation)

        # Simulate and score the explanation.
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=1,
                prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
                # prompt_format=PromptFormat.HARMONY_V4
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        results_json["scores"].append(scored_simulation.get_preferred_score())

        with open(filename, "w") as f:
            json.dump(results_json, f)

    return

import asyncio
asyncio.run(run())
