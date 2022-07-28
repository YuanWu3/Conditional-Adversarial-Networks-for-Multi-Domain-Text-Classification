# Conditional-Adversarial-Networks-for-Multi-Domain-Text-Classification
Implementation of "Conditional Adversarial Networks for Multi-Domain Text Classification" in Pytorch (Adapt-NLP at EACL 2020)

## Datasets
This folder contains the dataset in the same format as needed by our code.

## Requirements:
- Python 3.6
- PyTorch 0.4
- PyTorchNet
- scipy
- tqdm (for progress bar)

## Training
All the parameters are set as the same as parameters mentioned in the article. You can use the following commands to the tasks:

## MDTC experiments on the Amazon review dataset

cd code/

```
python train_man_exp1000.py --dataset prep-amazon --model mlp
```

## Citation
If you use this code for your research, consider citing:


    @article{wu2020dual,
      title={Conditional Adversarial Networks for Multi-Domain Text Classification},
      author={Wu, Yuan and Inkpen, Diana and El-Roby, Ahmed},
      booktitle={Proceedings of the Second Workshop on Domain Adaptation for NLP},
      pages={16--27},
      year={2021}
    }
