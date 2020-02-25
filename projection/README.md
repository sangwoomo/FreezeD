# Freeze D for SNGAN-projection

## Getting started

### Download datasets
```
# Download datasets in ./datasets directory with ImageFolder format (class: 000-999)
# Oxford Flower: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
# CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# Caltech-256: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
```

### Preprocess datasets
```
cd datasets
python make_image_list.py DATASET
cd ..
```

### Download pre-trained GAN models
```
# Download from https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi
# Save models in ./pretrained/sn_projection_128 directory (We use 850,000 iteration one)
```

### Pre-compute FID statistics
```
python source/inception/download.py --outfile=datasets/inception_model
python evaluations/calc_ref_stats.py --dataset DATASET --n_classes N_CLASSES --inception_model_path datasets/inception_model
```

### Run experiments
```
python finetune.py --data_dir=datasets/DATASET --config=configs/sn_projection_DATASET.yml --results_dir=results/DATASET_finetune --gpu 0
python finetune.py --data_dir=datasets/DATASET --config=configs/sn_projection_DATASET.yml --results_dir=results/DATASET_freeze --layer 3 --gpu 0
```

