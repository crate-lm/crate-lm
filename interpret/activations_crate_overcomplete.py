# In[1]:


import sys
import scipy
sys.path.append("..")

import os
import pickle
import torch
import tiktoken
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
import json

from tqdm import tqdm
from gpt import GPTConfig, GPT
from contextlib import nullcontext
from crate_overcomplete import CRATEConfig, CRATE
# from crate import CRATEConfig, CRATE

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# from neel.imports import *

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")

import os
import pickle
import torch
import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from gpt import GPTConfig, GPT
from contextlib import nullcontext

import torch.nn as nn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default="gpt-3L", help='model name')
parser.add_argument('--sae', type=str, default=None, help='sae')
parser.add_argument('--layer', type=str, default=None, help='sae')

args = parser.parse_args()
MODEL = args.model
SAE = args.sae # None, "1x", "4x", or "16x"

MORE_SAMPLES = False

# In[2]:


out_dir = '../out'
# n_layer = 1
# n_head = 8
# n_embd = 512
block_size = 1024
# bias = False
# dropout = 0
data_dir = '../data/pile/'

device = 'cuda'


# In[3]:


enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


# In[4]:


# split the data into chunks of block_size
data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
X = []
# batch_size = 8000 if not MORE_SAMPLES else 40000
batch_size = 8000
GPU_LIMIT = 40
# batch_id = 0

# starting_token_id = block_size * batch_size * batch_id
# ending_token_id = block_size * batch_size * (batch_id + 1)
# step_size = block_size
# for i in range(starting_token_id, ending_token_id, step_size):
#     X.append(torch.from_numpy(data[i:i+block_size].astype(np.int64)))
# len(X)

# Calculate the starting and ending token indices for random sampling
starting_token_id = block_size * batch_size
ending_token_id = block_size * batch_size
# random_indices = np.random.randint(0, len(data) - block_size, size=batch_size)
starting_indices = block_size * np.arange(batch_size)

# Extract the corresponding data segments and convert them to tensors
X = []
# for i in random_indices:
for i in starting_indices:
    X.append(torch.from_numpy(data[i:i+block_size].astype(np.int64)))

# Now, X contains random samples of data segments of size block_size


# In[5]:


activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# In[6]:


def load_model():
    # init from a model saved in a specific directory
    if MODEL == 'crate-1L':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-1l-overparam.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
        print("loading CRATE checkpoint")
    elif MODEL == 'crate-2L':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-2l-overparam.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
        print("loading CRATE checkpoint")
    elif MODEL == 'crate-3L':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-3l-overparam.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
        print("loading CRATE checkpoint")
    elif MODEL == 'crate-6L':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-6l-overparam.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
        print("loading CRATE checkpoint")
    elif MODEL == 'crate-12L-half':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-12l-overparam-300000.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
    elif MODEL == 'crate-12L-tenth':
        ckpt_path = os.path.join(out_dir, 'ckpt-crate-12l-overparam-60000.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        crateconf = CRATEConfig(**checkpoint['model_args'])
        model = CRATE(crateconf)
        print("loading CRATE checkpoint")
    elif MODEL == 'gpt-1L':
        ckpt_path = os.path.join(out_dir, 'ckpt-gpt-1l.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        print("loading GPT checkpoint")
    elif MODEL == 'gpt-2L':
        ckpt_path = os.path.join(out_dir, 'ckpt-gpt-2l.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        print("loading GPT checkpoint")
    elif MODEL == 'gpt-3L':
        ckpt_path = os.path.join(out_dir, 'ckpt-gpt-3l.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        print("loading GPT checkpoint")
    elif MODEL == 'gpt-6L':
        ckpt_path = os.path.join(out_dir, 'ckpt-gpt-6l.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        print("loading GPT checkpoint")
    elif MODEL == 'gpt-12L':
        ckpt_path = os.path.join(out_dir, 'ckpt-gpt-12l-60000.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        print("loading GPT checkpoint")
    state_dict = checkpoint['model']
    print(state_dict.keys())
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model


# In[7]:


def convert_to_cmap(clustered_feature_ids, dim=512):
    # clustered_feature_ids: [[4, 5], [3, 2, 0], ...
    #  -> [[4->1, 5->1], [3->2, 2->2, 0->2], ...], all other tokens get 0
    # convert to cmap: [1, 0, 2, 2, 1, 1, ...]

    # Initialize cmap with zeros
    cmap = [0] * dim

    # Assign unique numbers to each element in the sublists
    for cluster_id, sublist in enumerate(clustered_feature_ids, start=1):
        for feature_id in sublist:
            cmap[feature_id] = cluster_id

    return cmap

# use umap to reduce the dim of flattened_activation to 2 and plot it
def draw_umap(flattened_activation, clustered_feature_ids):
    from sklearn.decomposition import PCA
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Convert your clustered_feature_ids to a color map
    # Assuming convert_to_cmap is a function you've defined elsewhere
    cmap = convert_to_cmap(clustered_feature_ids, dim=flattened_activation.shape[1])

    # print("Reducing dimensionality with PCA...")
    # pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    # embedding = pca.fit_transform(flattened_activation.T)
    
    # use umap instead
    # print(len(sns.color_palette()))
    import umap.umap_ as umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(flattened_activation.T)

    sns.set(style='white', context='notebook', rc={'figure.figsize':(28,20)})
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette('hls', len(cmap))[x] for x in cmap],
        )
    plt.gca().set_aspect('equal', 'datalim')
    # plot the id of each feature
    for i in range(embedding.shape[0]):
        plt.text(embedding[i, 0], embedding[i, 1], str(i), fontsize=8)
    
    # draw a line between each meaningful feature pair
    # for pair in meaningful_feature_ids:
        # plt.plot(embedding[pair, 0], embedding[pair, 1], c='red')
    plt.title('UMAP projection of the Activation Vectors', fontsize=24)


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_meaningful_feature_ids(flattened_activation, threshold):
    # calculate pair-wise cosine similarity of each feature vector, those with > 0.9 similarity are considered meaningful
    # the final results will be a triangular matrix, with diagonal elements being 1, upper triangular elements being 0, and lower triangular elements being the cosine similarity
    
    flattened_activation = flattened_activation.T
    flattened_activation.to(device)
    # print(cos_sim(flattened_activation[:, 65], flattened_activation[:, 110]))
    cos_sim_mat = cos_sim(flattened_activation, flattened_activation)
    del flattened_activation
    
    meaningful_feature_ids = []
    # print("Getting meaningful feature ids...")
    for i in range(cos_sim_mat.shape[0]):
        for j in range(cos_sim_mat.shape[1]):
            if i != j and ([j, i] not in meaningful_feature_ids) and cos_sim_mat[i][j] > threshold:
                meaningful_feature_ids.append([i, j])
                
    # get a cluster using the meaningful_feature_ids
    clustered_feature_ids = []
    for i in range(len(meaningful_feature_ids)):
        feature_id = meaningful_feature_ids[i]
        if len(clustered_feature_ids) == 0:
            clustered_feature_ids.append([feature_id[0], feature_id[1]])
        else:
            for j in range(len(clustered_feature_ids)):
                if feature_id[0] in clustered_feature_ids[j] or feature_id[1] in clustered_feature_ids[j]:
                    clustered_feature_ids[j].append(feature_id[0])
                    clustered_feature_ids[j].append(feature_id[1])
                    break
                if j == len(clustered_feature_ids) - 1:
                    clustered_feature_ids.append([feature_id[0], feature_id[1]])
    clustered_feature_ids = [list(set(x)) for x in clustered_feature_ids]
                
    return clustered_feature_ids


# In[8]:


if SAE:
    import sae
    cfg = {
        "seed": 1, 
        "batch_size": 10,  # Number of samples we pass through THE LM 
        "seq_len": 1024,  # Length of each input sequence for the model
        "d_in": 512,  # Input dimension for the encoder model
        "d_sae": 512 * int(SAE.split("x")[0]),  # Dimensionality for the sparse autoencoder (SAE)
        "lr": 1.2e-3,  # This is low because Neel uses L2, and I think we should use mean squared error
        "l1_lambda": 1.6e-4,
        "dataset": "-",  # Name of the dataset to use
        "dataset_args": [],  # Any additional arguments for the dataset
        "dataset_kwargs": {"split": "train", "streaming": True}, 
        # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
        "beta1": 0.9,  # Adam beta1
        "beta2": 0.999,  # Adam beta2
        "num_tokens": int(2e12), # Number of tokens to train on 
        "test_set_batch_size": 1,
        "test_set_num_batches": 300,
        "wandb_mode_online_override": False, # Even if in testing, wandb online anyways
        "test_every": 500,
        "save_state_dict_every": lambda step: step%19000 == 1, # Disabled currently; used Mod 1 so this still saves immediately. Plus doesn't interfere with resampling (very often)
        "wandb_group": None,
        "resample_condition": "freq", # Choose nofire or freq
        "resample_sae_neurons_cutoff": lambda step: (1e-6 if step < 25_000 else 1e-7), # Maybe resample fewer later... only used if resample_condition == "nofire"
        "resample_mode": "anthropic", # Either "reinit" or "Anthropic"
        "anthropic_resample_batches": 20_000, # 200_000 # 32_000 // 100, # How many batches to go through when doing Anthropic reinit. Should be >=d_sae so there are always enough. Plus 
        "resample_sae_neurons_every": 205042759847598434752987523487239,
        "resample_sae_neurons_at": [10_000, 20_000] + torch.arange(50_000, 125_000, 25_000).tolist(),
        "dtype": torch.float32, 
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "activation_training_order": "shuffled", # Do we shuffle all MLP activations across all batch and sequence elements (Neel uses a buffer for this), using `"shuffled"`? Or do we order them (`"ordered"`)
        "buffer_size": 2**19, # Size of the buffer
        "buffer_device": "cuda:0", # Size of the buffer
        "testing": False,
        "delete_cache": False, # TODO make this parsed better, likely is just a string
        "sched_type": "cosine_warmup", # "cosine_annealing", # Mark as None if not using 
        "sched_epochs": 50*20, # Think that's right???
        "sched_lr_factor": 0.1, # This seems to help a little. But not THAT much, so tone down
        "sched_warmup_epochs": 50*20,
        "sched_finish": True,
        "resample_factor": 0.2, # 3.4 steps per second
        "bias_resample_factor": 0.0,
        "log_everything": False,
        "anthropic_resample_last": 7500,
        "l1_loss_form": "l1",
        "l2_loss_form": "normalize",
    }

    autoencoder = sae.SAE(cfg)


# In[9]:


def get_acts(block_name, layer_id):
    model = load_model()
    model.eval()
    
    if 'crate' in MODEL:
        model.transformer.h[layer_id].attn.y_hook.register_forward_hook(get_activation('attn2'))
    else:
        model.transformer.h[layer_id].attn.y_hook.register_forward_hook(get_activation('attn2'))
    if 'crate' in MODEL:
        model.transformer.h[layer_id].attn.c_attn.register_forward_hook(get_activation('attn'))
    else:
        model.transformer.h[layer_id].attn.query_hook.register_forward_hook(get_activation('attn'))
    if 'crate' in MODEL:
        model.transformer.h[layer_id].ista.relu.register_forward_hook(get_activation('ista')) # for CRATE
    else:
        model.transformer.h[layer_id].mlp.relu.register_forward_hook(get_activation('ista')) # for GPT
    
    # forward multiple times and aggregate the activations to avoid OOM
    aggregation = {"attn": [], "attn2": [], "ista": [], "ista_sae": []}
    print("forwarding multiple times and aggregate the activations to avoid OOM")
    for i in tqdm(range(0, len(X), GPU_LIMIT)):
        x = torch.stack(X[i:i+GPU_LIMIT]).to(device)
        output = model(x)
        aggregation['attn'].append(activation['attn'].cpu())
        aggregation['ista'].append(activation['ista'].cpu())
        aggregation['attn2'].append(activation['attn2'].cpu())
        # ista_act = activation['ista']
        # torch.save(ista_act, f'activations/ista_act_{i}.pt')
        # print(activation['ista'])
        
        # ****** SAE *******
        if SAE:
            dir_path = f"/home/ubuntu/nanogpt4crate/arthursae/weights/{MODEL.lower()}-{SAE}/{layer_id}"
            # get file path (file is the only file in the directory)
            file_name = os.listdir(dir_path)[0]
            path = os.path.join(dir_path, file_name)
            autoencoder.load_from_local(path=path)
            
            ista_act = activation['ista'].reshape(-1, activation['ista'].shape[2])
            decoded_activations, learned_activations = autoencoder(ista_act, return_mode="both")
            flattened_learned_activations = learned_activations.reshape(activation['ista'].shape[0], activation['ista'].shape[1], learned_activations.shape[-1])
            aggregation['ista_sae'].append(flattened_learned_activations.detach().cpu())
        # ****** SAE *******
        del x, output
    del model
    
    # concatenate the activations
    aggregation['attn'] = torch.cat(aggregation['attn'], dim=0)
    aggregation['attn2'] = torch.cat(aggregation['attn2'], dim=0)
    aggregation['ista'] = torch.cat(aggregation['ista'], dim=0)

    # ****** SAE *******
    if SAE:
        aggregation['ista_sae'] = torch.cat(aggregation['ista_sae'], dim=0)
        block_name += "_sae"
    # ****** SAE *******
    
    num_samples = aggregation[block_name].shape[0]
    num_tokens = aggregation[block_name].shape[1]
    num_features = aggregation[block_name].shape[2]
    
    # flatten both activation['ista'] and X
    print("aggregation[block_name].shape:", aggregation[block_name].shape)
    flattened_activation = aggregation[block_name].reshape(-1, aggregation[block_name].shape[2])
    # save the flattened_activation to a file
    print("flattened_activation.shape:", flattened_activation.shape)

    # now we get the tokens of top activations of each feature
    top_tokens = []
    top_token_act_values = []
    top_token_contexts = []
    original_top_token_context_lists = []
    top_token_context_act_values = []
    
    random_tokens = []
    random_token_act_values = []
    random_token_contexts = []
    original_random_token_context_lists = []
    random_token_context_act_values = []
    
    quantiles = [0, 0.5, 0.9, 0.99, 0.999, 1]
    quantiles = [1-quantile for quantile in quantiles]
    quantile_tokens = [[] for i in range(num_features)]
    quantile_token_act_values = [[] for i in range(num_features)]
    quantile_token_contexts = [[] for i in range(num_features)]
    original_quantile_token_context_lists = [[] for i in range(num_features)]
    quantile_token_context_act_values = [[] for i in range(num_features)]
    
    sample_ids = []
    token_ids = []
    # calculate mean, variance, skew and kurtosis of each feature
    means = []
    variances = []
    skews = []
    kurtosises = []
    
    # comment these when lots of samples are used
    clustered_feature_ids = []
    # clustered_feature_ids = get_meaningful_feature_ids(flattened_activation, threshold=0.80)
    # draw_umap(flattened_activation, clustered_feature_ids)
    
    flattened_X = torch.cat(X, dim=0)
    
    # flattened_sample_ids: [0, 0, 0, ..., 1, 1, 1, ...]
    flattened_sample_ids = torch.arange(num_samples).repeat_interleave(num_tokens).to(device)
    # flattened_token_ids: [0, 1, 2, ..., 0, 1, 2, ...]
    flattened_token_ids = torch.arange(num_tokens).repeat(num_samples).to(device)
    
    SPACE = "·"
    NEWLINE="↩"
    TAB = "→"

    for feature_id in tqdm(range(num_features)):
        means.append(torch.mean(flattened_activation[:, feature_id]).item())
        variances.append(torch.var(flattened_activation[:, feature_id]).item())
        skews.append(torch.mean((flattened_activation[:, feature_id] - means[-1]) ** 3).item())
        kurtosises.append(torch.mean((flattened_activation[:, feature_id] - means[-1]) ** 4).item())
        
        top_k = 50
        ordered_activations = torch.argsort(flattened_activation[:, feature_id], descending=True)

        top_activations = ordered_activations[:top_k].tolist()
        top_tokens.append([enc.decode([int(flattened_X[activation_id])]) for activation_id in top_activations])
        top_token_act_values.append([flattened_activation[activation_id, feature_id].item() for activation_id in top_activations])
        
        random_activations = np.random.randint(0, flattened_activation.shape[0], size=top_k)
        random_tokens.append([enc.decode([int(flattened_X[activation_id])]) for activation_id in random_activations])
        random_token_act_values.append([flattened_activation[activation_id, feature_id].item() for activation_id in random_activations])

        quantile_top_k = 20
        # randomly get activations from different quantiles
        quantile_activations = []
        for i in range(len(quantiles) - 1):
            start = int(ordered_activations.shape[0] * quantiles[i + 1])
            end = int(ordered_activations.shape[0] * quantiles[i])
            # sample_list is a list of activation ids in ordered_activations from idx start to end
            quantile_interval = ordered_activations[start:end]
            # print(quantile_interval)
            quantile_activations.append(np.random.choice(quantile_interval, size=quantile_top_k, replace=False).tolist())
            quantile_tokens[feature_id].append([enc.decode([int(flattened_X[activation_id])]) for activation_id in quantile_activations[i]])
            quantile_token_act_values[feature_id].append([flattened_activation[activation_id, feature_id].item() for activation_id in quantile_activations[i]])
            
        interval = 32
        top_token_context_list = []
        original_top_token_context_list = []
        top_token_context_act_value = []
        
        for activation_id in top_activations:
            # get the context of the token
            start = max(0, activation_id - interval)
            end = min(activation_id + interval, len(flattened_X))
            context = flattened_X[start:end].tolist()
            context_act_value = flattened_activation[start:end, feature_id].tolist()
            top_token_context_act_value.append(context_act_value)
            # replace special tokens
            context = [enc.decode([int(token)]) for token in context]
            original_top_token_context_list.append(context)
            context = [token.replace('\n', NEWLINE) for token in context]
            context = [token.replace('\t', TAB) for token in context]
            context = [token.replace(' ', SPACE) for token in context]
            top_token_context_list.append("".join(context))
        top_token_contexts.append(top_token_context_list)
        top_token_context_act_values.append(top_token_context_act_value)
        original_top_token_context_lists.append(original_top_token_context_list)
        
        random_token_context_list = []
        original_random_token_context_list = []
        random_token_context_act_value = []  
        for activation_id in random_activations:
            # get the context of the token
            start = max(0, activation_id - interval)
            end = min(activation_id + interval, len(flattened_X))
            context = flattened_X[start:end].tolist()
            context_act_value = flattened_activation[start:end, feature_id].tolist()
            random_token_context_act_value.append(context_act_value)
            # replace special tokens
            context = [enc.decode([int(token)]) for token in context]
            original_random_token_context_list.append(context)
            context = [token.replace('\n', NEWLINE) for token in context]
            context = [token.replace('\t', TAB) for token in context]
            context = [token.replace(' ', SPACE) for token in context]
            random_token_context_list.append("".join(context))
        random_token_contexts.append(random_token_context_list)
        random_token_context_act_values.append(random_token_context_act_value)
        original_random_token_context_lists.append(original_random_token_context_list)
        
        quantile_token_context_list = []
        original_quantile_token_context_list = []
        quantile_token_context_act_value = []
        for i in range(len(quantiles) - 1):
            # make the second shape of quantile_activations 5 (not 5 * 20)
            in_quantile_token_context_list = []
            in_original_quantile_token_context_list = []
            in_quantile_token_context_act_value = []
            for activation_id in quantile_activations[i]:
                # get the context of the token
                start = max(0, activation_id - interval)
                end = min(activation_id + interval, len(flattened_X))
                context = flattened_X[start:end].tolist()
                context_act_value = flattened_activation[start:end, feature_id].tolist()
                in_quantile_token_context_act_value.append(context_act_value)
                # replace special tokens
                context = [enc.decode([int(token)]) for token in context]
                in_original_quantile_token_context_list.append(context)
                context = [token.replace('\n', NEWLINE) for token in context]
                context = [token.replace('\t', TAB) for token in context]
                context = [token.replace(' ', SPACE) for token in context]
                in_quantile_token_context_list.append("".join(context))
            quantile_token_context_list.append(in_quantile_token_context_list)
            quantile_token_context_act_value.append(in_quantile_token_context_act_value)
            original_quantile_token_context_list.append(in_original_quantile_token_context_list)
        quantile_token_contexts[feature_id] = quantile_token_context_list
        quantile_token_context_act_values[feature_id] = quantile_token_context_act_value
        original_quantile_token_context_lists[feature_id] = original_quantile_token_context_list
            
        # these variables below don't contain the random activations
        sample_ids.append([flattened_sample_ids[activation_id].item() for activation_id in top_activations])
        token_ids.append([flattened_token_ids[activation_id].item() for activation_id in top_activations])
        
    return top_tokens, top_token_act_values, top_token_contexts, original_top_token_context_lists, top_token_context_act_values, \
        random_tokens, random_token_act_values, random_token_contexts, original_random_token_context_lists, random_token_context_act_values, \
        quantile_tokens, quantile_token_act_values, quantile_token_contexts, original_quantile_token_context_lists, quantile_token_context_act_values, \
        means, variances, skews, kurtosises, \
        sample_ids, token_ids, clustered_feature_ids, flattened_activation 
        
def save_json(layer_id, neuron_id, \
        original_top_token_context_list, top_token_context_act_value, \
        original_random_token_context_list, random_token_context_act_value, \
        original_quantile_token_context_list, quantile_token_context_act_value, quantile_token_act_value, \
        means, variances, skews, kurtosises):
    neurons = {}
    neurons['dataclass_name'] = 'NeuronRecord'
    neurons['neuron_id'] = {'dataclass_name': 'NeuronId', 'layer_index': layer_id, 'neuron_index': neuron_id}
    
    neurons['random_sample'] = []
    for i in range(len(original_random_token_context_list)):
        inner_json = {}
        inner_json['dataclass_name'] = 'ActivationRecord'
        inner_json['tokens'] = original_random_token_context_list[i]
        inner_json['activations'] = random_token_context_act_value[i]
        neurons['random_sample'].append(inner_json)
        
    neurons['random_sample_by_quantile'] = []
    for i in range(len(quantile_token_act_value)):
        quantile_json = []
        for j in range(len(original_quantile_token_context_list[i])):
            inner_json = {}
            inner_json['dataclass_name'] = 'ActivationRecord'
            inner_json['tokens'] = original_quantile_token_context_list[i][j]
            inner_json['activations'] = quantile_token_context_act_value[i][j]
            quantile_json.append(inner_json)
        neurons['random_sample_by_quantile'].append(quantile_json)
    
    neurons['most_positive_activation_records'] = []
    for i in range(len(original_top_token_context_list)):
        inner_json = {}
        inner_json['dataclass_name'] = 'ActivationRecord'
        inner_json['tokens'] = original_top_token_context_list[i]
        inner_json['activations'] = top_token_context_act_value[i]
        neurons['most_positive_activation_records'].append(inner_json)
    
    quantiles = []
    for i in range(len(quantile_token_act_value)):
        quantiles.append(sum(quantile_token_act_value[i]) / len(quantile_token_act_value[i]))
    # print(quantiles)
    neurons['quantile_boundaries'] = [None, quantiles[1], quantiles[2], quantiles[3], quantiles[4], None]
    # print(neurons['quantile_boundaries'])
    
    neurons['mean'] = means[neuron_id]
    neurons['variance'] = variances[neuron_id]
    neurons['skewness'] = skews[neuron_id]
    neurons['kurtosis'] = kurtosises[neuron_id]

    MODEL_PATH = MODEL if not SAE else f"{MODEL}_sae"
    if MORE_SAMPLES:
        MODEL_PATH = MODEL_PATH + "_moresamples"
    if SAE:
        MODEL_PATH = MODEL_PATH + f"_{SAE}"
    path = f"/home/ubuntu/nanogpt4crate/neurons/{block_name}/{MODEL_PATH}/{layer_id}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{neuron_id}.json", 'w') as f:
        json.dump(neurons, f)
        
# make a df
layer_id = int(args.layer)
block_name = 'ista'
top_tokens, top_token_act_values, top_token_contexts, original_top_token_context_lists, top_token_context_act_values, \
    random_tokens, random_token_act_values, random_token_contexts, original_random_token_context_lists, random_token_context_act_values, \
    quantile_tokens, quantile_token_act_values, quantile_token_contexts, original_quantile_token_context_lists, quantile_token_context_act_values, \
    means, variances, skews, kurtosises, \
    sample_ids, token_ids, clustered_feature_ids, flattened_activation = get_acts(block_name, layer_id)

for neuron_id in tqdm(range(len(original_top_token_context_lists))):
    save_json(layer_id, neuron_id, \
        original_top_token_context_lists[neuron_id], top_token_context_act_values[neuron_id], \
        original_random_token_context_lists[neuron_id], random_token_context_act_values[neuron_id], \
        original_quantile_token_context_lists[neuron_id], quantile_token_context_act_values[neuron_id], quantile_token_act_values[neuron_id], \
        means, variances, skews, kurtosises)

