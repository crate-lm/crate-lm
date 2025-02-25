# # config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# # launch as the following (e.g. in a screen session) and wait ~5 days:
# # $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# wandb_log = True
# wandb_project = 'crate'
# wandb_run_name='crate_base_60M'

# # these make the total batch size be ~0.5M
# # 60 batch size * 1024 block size * 4 gradaccum * 2 GPUs = 491,520
# batch_size = 60
# block_size = 1024
# gradient_accumulation_steps = 8 * 2

# # 'crate':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
# # 'crate-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
# # 'crate-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
# # 'crate-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

# n_layer=12
# n_head=12
# n_embd=768

# dtype = 'float16'

# # this makes total number of tokens be 300B
# max_iters = 600000
# lr_decay_iters = 600000

# # eval stuff
# eval_interval = 1000
# eval_iters = 200
# log_interval = 10

# # weight decay
# weight_decay = 0.1

# # optimizer
# learning_rate = 6e-4 # max learning rate
# warmup_iters = 2000

# init_from = 'resume'

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'crate'
wandb_run_name='crate-1L'

# config of CRATE-1Layer
n_layer=1
n_head=4
n_embd=512
dataset = 'pile'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1