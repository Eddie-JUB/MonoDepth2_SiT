# Monodepth2 implementation for SiT Dataset

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)
>

Especially implementation for SiT Dataset
> **SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots**
>
> [NeurIPS 2023 Dataset and Benchmark Track(Poster)]([https://github.com/SPALaboratory/SiT-Dataset](https://neurips.cc/virtual/2023/poster/73508))
> 
> Github: https://github.com/SPALaboratory/SiT-Dataset
>


## ⚙️ Setup

This repo runs in envs as bellow:
```shell

Ubuntu 18.04
python 3.6
torch 1.8.0+cu111
CUDA 10.1
cudnn 7.6.5.32-1+cuda10.1

```

For Docker user, you can use 1)Docker file or directrly pull 2)Docker image and run as you wish:

1) Dockerfile
```
# Build the Docker Image
docker build -t monodepth2_sit_custom:latest .

# Run the Docker Container
docker run -it --gpus all --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /home/your/path:/home/your/path \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --privileged --net=host \
    --name monodepth2_sit_custom monodepth2_sit_custom:latest /bin/bash

```



2) Docker image
```shell

# pull docker image
docker pull eedddyyyybae/monodepth2_pytorch_sit:latest

# run container at the beginning with options
docker run -it --gpus all --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /home/your/path:/home/your/path \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --privileged --net=host \
    --name monodepth2_sit eedddyyyybae/monodepth2_pytorch_sit:latest /bin/bash

# run after made container
docker exec -it monodepth2_sit /bin/bash

```




## 🖼️ Test demo images with pre-trained models




```shell

python test_simple.py --image_path image/or/folder/path --model_name your/model

# test sample image
python test_simple.py --image_path samples/0.png --model_name sit

```

<p align="center">
  <img src="demo.gif" alt="example input output gif" width="600" />
</p>





| `model_name`          | Training modality | KiTTi pretrained | Model resolution  | abs_rel. error |  sq_rel  | rmse | rmse_log |       a1 |       a2 |       a3 | 
|-----------------------|-------------------|------------------|-------------------|----------------|----------|------|----------|----------|----------|----------|
| [`mono_640x192`](https://drive.google.com/file/d/16xXSrC4ks-Nfr5iVwleELw-Au1wqqwou/view?usp=drive_link)          | Mono              | No | 640 x 192 | 0.227                 |  1.240  |   5.092  |   0.336  |   0.638  |   0.855  |   0.931  |
| [`mono_640x192`](https://drive.google.com/file/d/1Lkq1Wcd-lsV_fZyirdQrkWB469J_mclL/view?usp=drive_link)   | Mono     | Yes | 640 x 192   | 0.240  |   1.633  |   5.092  |   0.350  |    0.634  |   0.856  |  0.933  |
| [`mono_1024x320`](https://drive.google.com/file/d/1gELAaImRAgD0k2f8gck2N05NbFEDMiXr/view?usp=drive_link) | Mono     | Yes | 1024 x 320 | 0.251  |   1.619  |   5.110  |   0.362  |   0.626  |   0.848  |   0.924  |


## Data pre-process for SiT dataset

Utilize "tools_sit.ipynb", which include guides to pre-process SiT dataset to train and test for Monodepth2 incluiding:
- Clean up unnesessary files from SiT dataset for depth estimation and convert to kitti dataset format
- Depth Map GT generation
- Visualization. e.g. projection, demo video generation


## ⏳ Training with SiT Dataset
You can download full SiT dataset: 
https://github.com/SPALaboratory/SiT-Dataset

Before train dataest, default setting for training and validation expect png images to jpg with this command, which also deletes the raw SiT '.png' files:
```
find sit_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

or you can use option flag '--png' for training.
By default models and tensorboard event files are saved to ~/tmp/<model_name>. This can be changed with the --log_dir flag.
```

# options are avaliable for 'batch_size', 'num_workers', 'log_dir'
python train_sit.py --model_name mono_model --data_path path/to/dataset --split sit --dataset sit

```

## SiT Evaluation
To prepare the ground truth depth maps run:
```
python export_gt_depth_sit.py --data_path sit_data --split sit

python export_gt_depth_sit.py --data_path sit_data --split sit_benchmark

```
Asuming that you have placed the SiT dataset in the default location of ./sit_data/.

The following example command evaluates a model:

```
python evaluate_depth_sit.py --load_weights_folder your/model/path --eval_mono --eval_split sit --data_path your/data/path
```


