# Win-CCReID
This is the official PyTorch implementation of our paper "Win-Win by Competition: Auxiliary-Free Cloth-Changing Person Re-identification"
The code will be released after the paper is received.

## Overview
Our model achieves a win-win situation by enriching the identity (ID)- preserving information carried by the appearance and structure features while maintaining holistic efficiency. In detail, we build a hierarchical competitive strategy that progressively accumulates meticulous ID cues with discriminating feature extraction at the global, channel, and pixel levels during model inference. After the hierarchical discriminative clues mining for appearance and structure features, enhanced ID-preserving features are crosswise-integrated to reconstruct images to reduce intra-class variations. Finally, by combing with self- and cross- ID penalties, the ACID model is trained under a generative adversarial learning framework to reduce the discrepancy with real data distribution. 
 
<img width="1000" alt="image" src="https://user-images.githubusercontent.com/26376754/175821851-5abae014-4c41-48dd-8d58-2c719f3c2f50.png">

## Preparation
#### Requirements
- Python 3.6
- GPU memory >= 15G (fp32)
- GPU memory >= 10G (fp16/fp32)
- NumPy
- PyTorch 1.0+
- [Optional] APEX (fp16/fp32)

## Dataset
We support several datasets as
- PRCC
- LTCC
- VC-Cloth
- Celeb-ReID
- Celeb-Light 

You should run the data processing code to change your file structure first:
```
python prepare_data.py
```

The final file structure will be like:
```
├── Datasets/
│   ├── PRCC/
│       ├── train/
│       ├── query/
│       ├── gallery/
│   ├── LTCC/...

```


## How to train ACID 
You can use the commond in ```run_train.sh``` to train the model. All changeable hyperameters are inclued. 


## How to evalute ACID 
After getting the trained model, you can use the commond in ```run_test.sh``` to make the evaluation. You just need to modify the model path and the data path to your own path. 


