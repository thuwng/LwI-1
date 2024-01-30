# Dataset preparation:
Cifar10, Cifar100, Mnist, SVHN datasets can be automatically downloaded with torchvision.datasets; Tiny-Imagenet requires manual downloading.

# Experiments
    python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0

