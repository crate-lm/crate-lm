<!-- <p align="center">
    <img src="./assets/digirl-logo-text.png" alt="logo" width="20%">
</p>
-->


<h3 align="center">
Improving Neuron-level Interpretability with White-box Language Models 
<br>
<b>To Appear at CPAL 2025, Oral</b>

</h3>


<p align="center">
| <a href="https://crate-lm.github.io/"><b>Website</b></a> | <a href="https://arxiv.org/abs/2410.16443"><b>Paper</b></a> |
</p>

---

Research Code for preprint "Improving Neuron-level Interpretability with White-box Language Models".

[Hao Bai*](https://jackgethome.com), [Yi Ma](https://people.eecs.berkeley.edu/~yima/)<br>
UC Berkeley, UIUC, HKU
<br>
*Work done at UC Berkeley

# The CRATE Language Model

## Pre-training

Download the [Uncopyrighted Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) dataset to `data`. Then run `run_pretrain.sh` to pre-train the CRATE language model. You can also get the 12L model we pre-trained [here](https://huggingface.co/JackBAI/CRATE-GPT-12L-Pile-600000steps).

## Performance Evaluation

Download the datasets you want to evaluate the CRATE language model. For example, you can get lambada, wikitext-2, openwebtext, and wikitext-103 from Huggingface datasets. After obtaining these datasets, just run `run_eval.sh` to evaluate the performance of the CRATE language model on downstream tasks.

## Interpretability Evaluation

You should first install the automated interpretability evaluation tools from OpenAI. I accomodated the code from only supporting OpenAI checkpoint to any model on HuggingFace you want to use for evaluation. You can find the code in `automated-interpretability` folder.

Then you should get the activations from the CRATE language model. You can use the `./interpret/activations_crate_overcomplete.py` script to get the activations. The activations will be saved in the `neurons` folder. Then use `neurons/eval.py` to aggregate the interpretability of the CRATE language model.

You can choose to evaluate the neuron-level interpretability with either OpenAI or Anthropic metric. You need to also change `./automated-interpretability/neuron_explainer/activations/activations.py` line 160 and 197 when you change from OpenAI to Anthropic metric.

# Sparse Auto-encoder

## Training

We recommend using SAE Lens (https://github.com/jbloomAus/SAELens) or SAE repo from [Arthur Conmy](https://github.com/ArthurConmy/sae/tree/main). This study was done before SAE Lens was proposed, so we used the SAE repo from Arthur Conmy (located at `./arthursae`).

Before training SAE, you need to install [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) from Neel Nanda. I taylored the repo to support the CRATE language model. You need to first transform the nanogpt model to transformer_lens model (`./TransformerLens/demos/nanogpt_to_transformer_lens.ipynb`). You can find the taylored code in `transformer-lens` folder. You can run the SAE training scripts after installing TransformerLens.

## Evaluation

After saving the trained model, you should output the feature activations in a folder and use the same evaluation method mentioned above to evaluate the neuron-level interpretability. Also use the `./interpret/activations_crate_overcomplete.py` script to get the feature activations.

## Inspection

You can use `arthursae/inspect_sae.ipynb` to inspect the trained SAE model.
