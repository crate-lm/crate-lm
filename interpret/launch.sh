# CUDA_VISIBLE_DEVICES=7 nohup python activations_crate_overcomplete_anthropic.py --model='gpt-1L' --layer=0 --sae='16x' > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=7 nohup python activations_crate_overcomplete_anthropic.py --model='crate-6L' --layer=0 > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=7 nohup python activations_crate_overcomplete_anthropic.py --model='crate-6L' --layer=1 > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=6 nohup python activations_crate_overcomplete_anthropic.py --model='crate-6L' --layer=2 > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=6 nohup python activations_crate_overcomplete_anthropic.py --model='crate-6L' --layer=3 > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=2 nohup python activations_crate_overcomplete.py --model='crate-12L-full' --layer=2 > nohup_crate.out &
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-quarter' --layer=0
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-quarter' --layer=1
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-quarter' --layer=2
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-quarter' --layer=3
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-quarter' --layer=4
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=0
# CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=1
CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=2
CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=3
CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=4
CUDA_VISIBLE_DEVICES=2 python activations_crate_overcomplete.py --model='crate-12L-tenth' --layer=5
