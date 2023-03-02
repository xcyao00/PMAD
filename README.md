# [One-for-All: Proposal Masked Cross-Class Anomaly Detection](https://arxiv.org/abs/2106.08254)

PyTorch implementation and pretrained models for AAAI2023 paper, One-for-All: Proposal Masked Cross-Class Anomaly Detection.


---


## Download Pretrained Weights and Models


Download checkpoints that are **self-supervised pretrained** on ImageNet-22k and **then used for fine-tuning** on MVTecAD dataset:
- ViT-base-16: [beit_base_patch16_224_pt22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth)
- ViT-large-16: [beit_large_patch16_224_pt22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth)

Download pretrained visual tokenizer(discrite VAE) from: [encoder](https://cdn.openai.com/dall-e/encoder.pkl), [decoder](https://cdn.openai.com/dall-e/encoder.pkl), and put them to the directory ``weights/tokenizer``.

Or download pretrained ViT visual tokenizer from: [vit_tokenizer](https://pan.baidu.com/s/15ySQAeBLhr3HYe_rECVqkQ) (extract code: hxcc), and put them to the directory ``weights/tokenizer``.

Download offline generated prototype feature from: [there](https://pan.baidu.com/s/1ewBLEp2gKfGU0kVGRupifw) (extract code: hxcc), and put it to the directory ``weights/prototypes``.

Download pretrained protoflow from: [there](https://pan.baidu.com/s/1tDg_lcAyRQfQikHOKt_h2w) (extract code: hxcc), and put it to the directory ``weights/protoflow``.


## Setup
Install all packages with this command:
```
$ python3 -m pip install -U -r requirements.txt
```

Download MVTecAD dataset from [there](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/), put it to the directory ``data/mvtec_anomaly_detection``, and then run following code to convert this dataset to ImageNet format.

```
python setup_train_dataset.py --data_path /path/to/dataset
```
This script will create a ImageNet format dataset for training at the ``data/Mvtec-ImageNet`` directory. Then please download [foreground masks](https://pan.baidu.com/s/1KrTAfe3QSvJuFPfppNAY3g) (extract code: hxcc), and put it to the directory ``data/Mvtec-ImageNet/fg_mask``.

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
You can download trained ``ViT-base-16`` models for multi-class setting: [multi-classes-model](https://pan.baidu.com/s/1zfPSFzLp6-QLb41nFk5gYA), the trained models are provided in [Download Pretrained Weights and Models](#download-pretrained-weights-and-models) section. You can download trained ``ViT-base-16`` model for cross-class setting from: [objects-to-textures](https://pan.baidu.com/s/12uIIv-_YSzthvgCLwjCiNQ) and [textures-to-objects](https://pan.baidu.com/s/1quth2urv-4rMmCdvP8GMFA).

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

