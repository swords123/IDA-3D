python -m torch.distributed.launch --nproc_per_node=2 tools/test_net.py --config-file self_exp/exp_1.yaml --ckpt self_exp/exp_1/model_0062500.pth
