# Freeze D: A Simple Baseline for Fine-tuning GANs

Code for "Freeze Discriminator: A Simple Baseline for Fine-tuning GANs".

The code is heavily based on the [StyleGAN-pytorch](https://github.com/rosinality/style-based-gan-pytorch) and [SNGAN-projection-chainer](https://github.com/pfnet-research/sngan_projection) codes.

See `stylegan` and `projection` directory for StyleGAN and SNGAN-projection experiments, respectively.

### Generated samples

Generated samples over fine-tuning FFHQ-pretrained StyleGAN

<img src="./resources/cat_trend.gif" width="384"> &nbsp; <img src="./resources/dog_trend.gif" width="384">


### More generated samples (StyleGAN)

Generated samples on [Animal Face](https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/exp5.html) and [Anime Face](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) datasets

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

Comparison of fine-tuning (left) and freeze D (right)

<img src="./resources/projection/flower_base.png" width="384"> &nbsp; <img src="./resources/projection/flower_freeze.png" width="384">

<img src="./resources/projection/cub_base.png" width="384"> &nbsp; <img src="./resources/projection/cub_freeze.png" width="384">

<img src="./resources/projection/caltech_base.png" width="384"> &nbsp; <img src="./resources/projection/caltech_freeze.png" width="384">
