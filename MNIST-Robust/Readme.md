## PGD adversarial training of the Small-CNN on the MNIST
### Usage
```
python PGD_CNN_WNLL.py
```

### Attack the trained robust
```
python Attack_CNN_WNLL_PGD.py --method ifgsm
```
The method can be fgsm, ifgsm, and cw
