# MAE-TAD: Towards Robust End-to-End Temporal Action Detection via Semantic Reconstruction

This repository contains the official implementation of the paper MAE-TAD: MAE-TAD: Towards Robust End-to-End Temporal Action Detection via Semantic Reconstruction.

![MAE-TAD Model](assets/model.jpg)

Several comments are remained.


# Getting Started

## Installation
```bash
cd util
python setup.py # build NMS
cd ..
```

## Prepare Dataset
We follow [ActionFormer](https://github.com/happyharrycn/actionformer_release) repository for preparing datasets including THUMOS14, ActivityNet v1.3, and EpicKitchens.

Use `scripts/make_feature_info.py` to generate feature information for each dataset.


<!-- ### THUMOS14 -->


## Training
To train the TE-TAD model on the THUMOS14 dataset, execute the following command:
```bash
python main.py --c configs/thumos14.yaml --output_dir logs/thumos14
```
## Evaluation
To evaluate the trained model and obtain performance metrics, use the following command structure:
```bash
python main.py --eval --c configs/thumos14.yaml --output_dir logs/thumos14
```
