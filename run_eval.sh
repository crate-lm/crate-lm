# CUDA_VISIBLE_DEVICES=3 python train.py config/eval_pile.py
CUDA_VISIBLE_DEVICES=0 python train.py config/eval_openwebtext.py
CUDA_VISIBLE_DEVICES=0 python train.py config/eval_lambada.py
CUDA_VISIBLE_DEVICES=0 python train.py config/eval_wikitext.py
CUDA_VISIBLE_DEVICES=0 python train.py config/eval_ptb.py
