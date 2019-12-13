# Text Triplet Loss using tensorflow
This repository provides a CNN implementation of Triplet Loss on Text data.

## Requirements
1. Python3
2. Tensorflow
3. Numpy
4. Fasttext

## Setup
Run setup.sh to setup required library and dependencies.
```buildoutcfg
bash setup.sh
```
Download pre-trained model from following Google Drive repository. 
```buildoutcfg
https://drive.google.com/file/d/1VL6BThgkRzfSjD-WtfukSvKPiwOZ-4a7/view?usp=sharing
```
Unzip and paste ```./model_triplets/``` directory to the project root directory.

## Pre-Trained Model:
Directory structure for pre-trained model should be as follows:
```buildoutcfg
1. ./model_triplets/checkpoint
2. ./model_triplets/model.ckpt.data-00000-of-00001
3. ./model_triplets/model.ckpt.index
4. ./model_triplets/model.ckpt.meta
5. ./model_triplets/vocab
```
## Training:
Run ```train_triplets.py ``` to train model.
```buildoutcfg
python train_triplets.py
```

## Prediction:
```buildoutcfg
python predict.py "how to become a data scientist?"
```
Use ```Prediction.ipynb``` to get more detailed prediction code.
