# DNN-DataDependentActivation
This repository consists PyTorch code for deep neural networks with graph interpolating function as output activation function

## External dependency: pyflann (https://github.com/primetang/pyflann)
Place the pyflann library in your current directory to replace the pyflann folder

### Cifar10-Natural
Code for reproducing results of naturally trained ResNets on the Cifar10

### Cifar10-Robust
Code for reproducing results of PGD adversarial training for ResNets on the Cifar10

### MNIST-Robust
Code for reproducing results of PGD adversarial training for Small-CNN on the MNIST

If you find this work useful and use it on you own research, please cite our [paper](https://papers.nips.cc/paper/7355-deep-neural-nets-with-interpolating-function-as-output-activation.pdf)

```
@incollection{NIPS2018_7355,
title = {Deep Neural Nets with Interpolating Function as Output Activation},
author = {Wang, Bao and Luo, Xiyang and Li, Zhen and Zhu, Wei and Shi, Zuoqiang and Osher, Stanley},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {743--753},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7355-deep-neural-nets-with-interpolating-function-as-output-activation.pdf}
}
```

And the longer version is available at

```
@ARTICLE{Wang:2019Interpolation,
       author = {{B. Wang and S. Osher},
        title = "{Graph Interpolating Activation Improves Both Natural and Robust Accuracies in Data-Efficient Deep Learning}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Jul",
          eid = {arXiv:1907.06800},
        pages = {arXiv:1907.06800},
archivePrefix = {arXiv},
       eprint = {},
 primaryClass = {stat.ML}
}
```

## Dependence
PyTorch 0.4.1
