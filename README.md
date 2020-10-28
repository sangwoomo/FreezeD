# FreezeD: a Simple Baseline for Fine-tuning GANs

**Update (2020/10/28)**

Release [checkpoints](https://drive.google.com/drive/folders/140y2e80koKA_URy6cNChpK4LKqGjWnv0) of StyleGAN fine-tuned on cat and dog datasets.

**Update (2020/04/06)**

Current code evaluates FID scores with `inception.train()` mode. Fixing it to `inception.eval()` may degrade the overall scores (both competitors and ours; hence the trend does not change). Thanks to @jychoi118 ([Issue #3](https://github.com/sangwoomo/FreezeD/issues/3)) for reporting this.

---

Official code for ["**Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs**"](https://arxiv.org/abs/2002.10964) (CVPRW 2020).

The code is heavily based on the [StyleGAN-pytorch](https://github.com/rosinality/style-based-gan-pytorch) and [SNGAN-projection-chainer](https://github.com/pfnet-research/sngan_projection) codes.

See `stylegan` and `projection` directory for StyleGAN and SNGAN-projection experiments, respectively.

**Note:** There is a bug in PyTorch 1.4.0, hence one should use `torch>=1.5.0` or `torch<=1.3.0`. See Issue [#1](https://github.com/sangwoomo/FreezeD/issues/1).

## Generated samples

Generated samples over fine-tuning FFHQ-pretrained StyleGAN

<img src="./resources/cat_trend.gif" width="384"> &nbsp; <img src="./resources/dog_trend.gif" width="384">


### More generated samples (StyleGAN)

Generated samples under [Animal Face](https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/exp5.html) and [Anime Face](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) datasets

<img src="./resources/stylegan/original.png" width="256"> &nbsp; <img src="./resources/stylegan/bear.png" width="256"> &nbsp; <img src="./resources/stylegan/cat.png" width="256">

<img src="./resources/stylegan/chicken.png" width="256"> &nbsp; <img src="./resources/stylegan/cow.png" width="256"> &nbsp; <img src="./resources/stylegan/deer.png" width="256">

<img src="./resources/stylegan/dog.png" width="256"> &nbsp; <img src="./resources/stylegan/duck.png" width="256"> &nbsp; <img src="./resources/stylegan/eagle.png" width="256">

<img src="./resources/stylegan/elephant.png" width="256"> &nbsp; <img src="./resources/stylegan/human.png" width="256"> &nbsp; <img src="./resources/stylegan/lion.png" width="256">

<img src="./resources/stylegan/monkey.png" width="256"> &nbsp; <img src="./resources/stylegan/mouse.png" width="256"> &nbsp; <img src="./resources/stylegan/panda.png" width="256">

<img src="./resources/stylegan/pigeon.png" width="256"> &nbsp; <img src="./resources/stylegan/pig.png" width="256"> &nbsp; <img src="./resources/stylegan/rabbit.png" width="256">

<img src="./resources/stylegan/sheep.png" width="256"> &nbsp; <img src="./resources/stylegan/tiger.png" width="256"> &nbsp; <img src="./resources/stylegan/wolf.png" width="256">

<img src="./resources/stylegan/miku.png" width="256"> &nbsp; <img src="./resources/stylegan/sakura.png" width="256"> &nbsp; <img src="./resources/stylegan/haruhi.png" width="256">

<img src="./resources/stylegan/fate.png" width="256"> &nbsp; <img src="./resources/stylegan/nanoha.png" width="256"> &nbsp; <img src="./resources/stylegan/lelouch.png" width="256">

<img src="./resources/stylegan/mio.png" width="256"> &nbsp; <img src="./resources/stylegan/yuki.png" width="256"> &nbsp; <img src="./resources/stylegan/shana.png" width="256">


### More generated samples (SNGAN-projection)

Comparison of fine-tuning (left) and freeze D (right) under [Oxford Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) datasets

Freeze D generates more class-consistent results (see row 2, 8 of Oxford Flower)

<img src="./resources/projection/flower_base.png" width="384"> &nbsp; <img src="./resources/projection/flower_freeze.png" width="384">

<img src="./resources/projection/cub_base.png" width="384"> &nbsp; <img src="./resources/projection/cub_freeze.png" width="384">

<img src="./resources/projection/caltech_base.png" width="384"> &nbsp; <img src="./resources/projection/caltech_freeze.png" width="384">


## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{
    mo2020freeze,
    title={Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs},
    author={Mo, Sangwoo and Cho, Minsu and Shin, Jinwoo},
    booktitle = {CVPR AI for Content Creation Workshop},
    year={2020},
}
```
