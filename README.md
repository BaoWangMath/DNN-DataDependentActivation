# DNN-DataDependentActivation
Code for the paper: https://arxiv.org/pdf/1802.00168.pdf

## External dependency: pyflann (https://github.com/primetang/pyflann)
Place the pyflann library to your current directory

## Usage
### Step 1. Train the deep neural nets with softmax and WNLL activation functions
python TrainStandardDNN.py
python TrainWNLLDNN.py

### Step 2. Attack the trained deep neural nets
python Fool_StandardDNN.py -method fgsm -epsilon 0.02

python Fool_WNLLDNN.py -method fgsm -epsilon 0.02

The method and epsilon are adjustable, where we support fgsm, ifgsm, cwl2 attacks
