# COIN

## Description
The key code of our work "Improving Fine-tuning of Self-supervised Models with Contrastive Initialization".

## Installation
The important packages are as follows:
```python
# The main packages:
# pytorch=1.9.0
# torchvision=0.10.0
# pytorch-lightning==1.3.8
# tensorboard==2.4.1
# tensorboard==2.4.1

# Additional package for ViT:
# timm==0.4.12

# Additional package for calculating S_Dbw score
# s-dbw==0.4.0
```

Besides, you can update your environment using the provided `env.yaml`, just like this:

`conda env update --file env.yaml`

> We do not recommend this method because it contains some redundant packages.

## Instructions
We provide the implementation on CIFAR-10 and the bash file `run_fine_tuning_on_cifar10.sh`. You can easily run the code just like this:

`bash run_fine_tuning_on_cifar10.sh`

> If you need to modify the hyperparameters, please refer to the contents of the main program file and the bash file.

## Results
Furthermore, we provide two log files from running this code twice `run1.log` and `run2.log`. The results are shown in the following table:

| Runs | Top-1 Acc. | Total Loss |
| :------: | :------:  | :------: |
| 1 | 97.8799% | 0.106887 |
| 2 | 97.8799% | 0.103798 |

These results are basically the same as our reported top-1 accuracy on CIFAR-10 of **97.88%**, which show the reproducibility of our work.
