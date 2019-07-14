## PGD adversarial training of the ResNets on the Cifar10
### Usage
```
python main_PGD_ResNet_WNLL.py
```

### Attack the trained robust
```
python Attack_ResNet_WNLL_PGD.py --method ifgsm
```
The method can be fgsm, ifgsm, and cw
