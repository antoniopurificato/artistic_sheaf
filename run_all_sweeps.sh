export CUDA_VISIBLE_DEVICES=2

python3 main.py --sweep True --params_sweep batch_size
python3 main.py --sweep True --params_sweep lr

# python3 main.py --sweep True --params_sweep optimizer
python3 main.py --sweep True --params_sweep finetune_layers
python3 main.py --sweep True --params_sweep sheaf_layers

# python3 main.py --sweep True --params_sweep alpha
# python3 main.py --sweep True --params_sweep weights_components
# python3 main.py --sweep True --params_sweep weights_kl_vs_clip
# python3 main.py --sweep True --params_sweep w_clip_vs_mask 

# python3 main.py --sweep True --params_sweep test
# python3 main.py --sweep True --params_sweep out_proj
# python3 main.py --sweep True --params_sweep _laplacian_heat_kernel

# python3 main.py --dataset Hertziana
# python3 main.py --dataset Wikidataset
# python3 main.py --dataset SemArt


