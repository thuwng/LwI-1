# Dataset preparation:
Cifar-10, Cifar-100, MNIST, SVHN datasets can be automatically downloaded with torchvision.datasets; Tiny-Imagenet requires manual downloading.

# Experiments
To run on Cifar-100, ResNet32, excute:

    python main_incremental.py --dataset cifar100 --network 'resnet32' --nepochs 200 --learning_rate 1e-3 --ensemble_step 0.70 --ensemble_step_diff 0.93 --reg 0.01 --momentum 0.9

To run on Cifar100, ResNet18, excute:

    python main_incremental.py --dataset cifar100 --network 'resnet18' --nepochs 200 --learning_rate 1e-3 --ensemble_step 0.69 --ensemble_step_diff 0.90 --reg 0.01 --momentum 0.9

To run on Tiny-Imagenet, ResNet32, excute:

    python main_incremental.py --dataset imagenet_256 --network 'resnet32' --nepochs 200 --learning_rate 1e-3 --ensemble_step 0.71 --ensemble_step_diff 0.94 --reg 0.01 --momentum 0.0

To run on Tiny-Imagenet, ResNet18, excute:

    python main_incremental.py --dataset imagenet_256 --network 'resnet18' --nepochs 200 --learning_rate 1e-3 --ensemble_step 0.85 --ensemble_step_diff 0.65 --reg 0.01 --momentum 0.0

# Reference
The code structure is based on the code in [FACIL](https://github.com/mmasana/FACIL) and [otfusion](https://github.com/sidak/otfusion).
