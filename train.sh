python -m torch.distributed.launch --nproc_per_node=2 --master_port=6001 tools/train_net.py --config-file self_exp/exp_1.yaml
