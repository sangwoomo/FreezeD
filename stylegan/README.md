# Freeze D for StyleGAN

## Getting started

### Download datasets
```
Animal Face: https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/AnimalFace.zip
Anime Face: http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
```

### Preprocess datasets
```
python prepare_data.py --out dataset/DATASET_lmdb --n_worker 8 dataset/DATASET
```

### Download pre-traind GAN models
```
# Download from https://drive.google.com/file/d/1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ/view
# Save model in ./checkpoint directory
```

### Pre-compute FID activations
```
python precompute_acts.py --dataset DATASET
```

### Run experiments
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --name DATASET_finetune --mixing --loss r1 --sched --dataset DATASET
CUDA_VISIBLE_DEVICES=1 python finetune.py --name DATASET_freezeD --mixing --loss r1 --sched --dataset DATASET --freeze_D --feature_loc 3
# Note that feature_loc = 7 - layer_num
```