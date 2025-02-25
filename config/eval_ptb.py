# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters

dataset = "ptb"
eval_interval = 3000
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True
wandb_log = False
# n_layer=12
# n_head=8
# n_embed=512

batch_size = 4
block_size = 1024
dtype = 'float16'

init_from = 'gpt2'
