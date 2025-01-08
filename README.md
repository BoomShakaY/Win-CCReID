# Win-CCReID
This is the official PyTorch implementation of our paper "Win-Win by Competition: Auxiliary-Free Cloth-Changing Person Re-identification" 

The code has been released and it still has several bugs. Thanks for waiting.  
Since I struggled to find only some parts of the code and re-write the other missing parts. 
You may refer to the specific part of the code to get what you want, but running the whole project still needs effort to debug.  
I will try to fix them in the next version, thanks again.

If it offers you some help, please consider citing our paper, much appreciated!
```
@article{yang2023win,
  title={Win-win by competition: Auxiliary-free cloth-changing person re-identification},
  author={Yang, Zhengwei and Zhong, Xian and Zhong, Zhun and Liu, Hong and Wang, Zheng and Satoh, Shin'Ichi},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={2985--2999},
  year={2023},
  publisher={IEEE},
  doi={10.1109/TIP.2023.3277389}
}
```

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

Download the above datasets through their official website first.  

You may run the data processing code in  ```prepare_dataset.sh```  to change your file structure first:  
Note to modify the dataset path to your own dataset path.

The final file structure will be like:
```
├── Datasets/
│   ├── PRCC/
│       ├── train/
|           ├── A/
|           ├── B/
|           ├── C/
│       ├── test/
│       ├── val/
│   ├── LTCC_ReID/
│       ├──pytorch
│          ├── train/
│          ├── query/
│          ├── gallery/
│   ├── VC-Clothes/
│       ├──pytorch
│          ├── train/
│          ├── query/
│          ├── gallery/
│   ├── ...

```

## Before you train ACID
Please train the teacher model first. Please check the [person re-id baseline](https://github.com/layumi/Person_reID_baseline_pytorch) repository to train a teacher model, then copy and put it in the `./models`.
```
├── models/
│   ├── prcc/                   /* teacher model for prcc datast
│       ├── net_last.pth        /* model file
│       ├── ...
```

## How to train ACID 
1. Setup the yaml file. Check out `configs/CA_*.yaml`. Change the data_root field to the path of your prepared folder-based dataset, e.g. `../LTCC/pytorch`. As well as the teacher model name you saved before. 

2. Run the training script. You can use the command in ```run_train.sh``` to train the model.

 ```python train.py --config configs/CA_%target_dataset%.yaml --model_name ACID_Default ```

 Intermediate image outputs and model binary files are saved in `outputs/ACID_Default`.

3. Check the loss log
```
 tensorboard --logdir logs/ACID_Default
```

## How to evalute ACID 
After getting the trained model, you can use the commond in ```run_test.sh``` to make the evaluation. You just need to modify the model path and the data path to your own path. 


