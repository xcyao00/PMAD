# [One-for-All: Proposal Masked Cross-Class Anomaly Detection](https://arxiv.org/abs/2106.08254)

PyTorch implementation and pretrained models for AAAI2023 paper, One-for-All: Proposal Masked Cross-Class Anomaly Detection.


---


## Download Pretrained Weights and Models

We provide four weights trained on MVTecAD dataset. The models were trained with 224x224 resolution.

- `ViT-base-16`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `ViT-large-16`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)

Download checkpoints that are **self-supervised pretrained** on ImageNet-22k and **then intermediate fine-tuned** on MVTecAD dataset:
- ViT-base-16: [beit_base_patch16_224_pt22k_ft22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth)
- ViT-large-16: [beit_large_patch16_224_pt22k_ft22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)

Download pretrained visual tokenizer(discrite VAE) from: [encoder](https://cdn.openai.com/dall-e/encoder.pkl), [decoder](https://cdn.openai.com/dall-e/encoder.pkl), and put them to the directory ``weights/tokenizer``.

Download offline generated prototype feature from: [there](https://cdn.openai.com/dall-e/encoder.pkl), and put it to the directory ``weights/prototypes``.

Download pretrained protoflow from: [there](https://cdn.openai.com/dall-e/encoder.pkl), and put it to the directory ``weights/protoflow``.


## Setup
Install all packages with this command:
```
$ python3 -m pip install -U -r requirements.txt
```

Download MVTecAD dataset from [there](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/), put it to the directory ``data/mvtec_anomaly_detection``, and then run following code to convert this dataset to ImageNet format.

```
python setup_train_dataset.py --data_path /path/to/dataset
```
This script will create a ImageNet format dataset for training at the ``data/Mvtec-ImageNet`` directory. Then please download [foreground masks](https://cdn.openai.com/dall-e/encoder.pkl), and put it to the directory ``data/Mvtec-ImageNet/fg_mask``.

## Training

Run code for training MVTecAD dataset.
```
bash scripts/train_multi_class.sh  // training for multi-class setting
bash scripts/train_cross_class.sh  // training for cross-class setting
```
For cross-class setting objects-to-textures, please set ``--from_obj_to_texture`` in ``train_cross_class.sh``. If not setted, the code will run cross-class setting textures-to-objects.


## Testing

Run code for testing MVTecAD dataset.
```
bash scripts/test_multi_class.sh  // testing for multi-class setting
bash scripts/test_multi_class.sh  // testing for multi-class setting
```
You can download trained ``ViT-base-16`` models for multi-class setting: [multi-classes-model](output_dir/vit_base_16_checkpoint_962_955.pth), the trained models are provided in [Download Pretrained Weights and Models](#download-pretrained-weights-and-models) section. You can download trained ``ViT-base-16`` model for cross-class setting from: [objects-to-textures](output_dir/vit_base_16_obj_to_texture/checkpoint_976_931.pth) and [textures-to-objects](output_dir/vit_base_16_texture_to_obj/checkpoint_844_894.pth).

We summarize the validation results as follows.

Multi-Class Setting:

| Category | Image/Pixel AUC | Category | Image/Pixel AUC | Category | Image/Pixel AUC |
|:------------:|:--------:|:----------:|:-----:|:-----:|:-------:|
| Carpet | 0.999/0.988 | Bottle | 1.000/0.978 | Pill | 0.965/0.952 |
| Grid | 0.982/0.962 | Cable | 0.975/0.963 | Screw | 0.807/0.954 |
| Leather | 1.000/0.990 | Capsule | 0.912/0.962 | Toothbrush | 0.894/0.980 |
| Tile | 1.000/0.956 | Hazelnut | 1.000/0.980 | Transistor | 0.963/0.940 |
| Wood | 1.000/0.908 | Metal nut | 1.000/0.888 | Zipper | 0.967/0.942 |
| Mean | 0.964/0.956 | 
---
Cross-Class Setting(objects-to-textures):

| Category | Image/Pixel AUC | 
|:------------:|:--------:|
| Carpet | 0.986/0.967 | 
| Grid | 0.901/0.913 | 
| Leather | 1.000/0.978 | 
| Tile | 0.998/0.935 | 
| Wood | 0.995/0.862 | 
| Mean | 0.976/0.931 |
---
Cross-Class Setting(textures-to-objects):

| Category | Image/Pixel AUC | Category | Image/Pixel AUC |
|:----------:|:-----:|:-----:|:-------:|
| Bottle | 0.977/0.932 | Pill | 0.805/0.860 |
| Cable | 0.893/0.940 | Screw | 0.580/0.911 |
| Capsule | 0.767/0.956 | Toothbrush | 0.917/0.965 |
| Hazelnut | 0.939/0.928 | Transistor | 0.834/0.801 |
| Metal nut | 0.787/0.716 | Zipper | 0.945/0.927 |
| Mean | 0.844/0.894 | 
---


## Citation

If you find this repository useful, please consider citing our work:
```
@article{PMAD,
      title={One-for-All: Proposal Masked Cross-Class Anomaly Detection}, 
      author={Xincheng Yao and Chongyang Zhang and Ruoqi Li and Jun Sun and Zhenyu Liu},
      year={2022},
      eprint={2206.08254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository and the [DALL-E](https://github.com/openai/DALL-E) repository.

