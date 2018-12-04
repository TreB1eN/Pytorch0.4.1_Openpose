# Pytorch0.4.1_Realtime\_Multi-Person\_Pose\_Estimation

This is an implementation of [Realtime Multi-Person Pose Estimation](https://arxiv.org/abs/1611.08050) with Pytorch.
The original project is <a href="https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation">here</a>. 

This repo is mainly based on the [Chainer Implementation](https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation) from DeNA.

This project is licensed under the terms of the [license](https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/LICENSE)

Some contributions are:

1. Switching the Chainer backbone to Pytorch
2. Pretrained models in pytorch are provided
3. Chinese comments to better understanding the method

## Content

1. [Converting caffe model](#convert-caffe-model-to-chainer-model)
2. [Testing](#test-using-the-trained-model)
3. [Training](#train-your-model-from-scratch)

## Test using the trained model

Download the following pretrained models to `models` folder

posenet.pth : [@Google Drive](https://drive.google.com/open?id=19AIYt2lez5V3x4wFVJvVvWwQpB8uoQp2)  [@One Drive](https://1drv.ms/u/s!AhMqVPD44cDOhxrwKTyv9yv3FRVq)

facenet.pth : [@Google Drive](https://drive.google.com/open?id=1zjv4fQt3Sd567VpesqAEO4JVsjB79xpZ)  [@One Drive](https://1drv.ms/u/s!AhMqVPD44cDOhxwu2Nmf1eXNmOXd)

handnet.pth : [@Google Drive](https://drive.google.com/open?id=1LdWngNbcamMJFAuRaqT45Iar2tA_eUXd)  [@One Drive](https://1drv.ms/u/s!AhMqVPD44cDOhxs1EIBYqksR6avn)

Execute the following command with the weight parameter file and the image file as arguments for estimating pose.
The resulting image will be saved as `result.jpg`.

```
python pose_detect.py models/posenet.pth -i data/football.jpg
```

<div align="center">
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/football.jpg" width="800" height="600">
&nbsp;
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/football_detected.jpg" width="800" height="600">
</div>



Similarly, execute the following command for face estimation.
The resulting image will be saved as `result.png`.

```
python face_detector.py models/facenet.pth -i data/face.png
```

<div align="center">
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/face.jpg" width="300">
&nbsp;
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/face_result.png" width="300">
</div>



Similarly, execute the following command for hand estimation.
The resulting image will be saved as `result.jpg`.

```
python hand_detector.py models/handnet.pth -i data/hand.jpg
```

<div align="center">
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/hand.jpg" width="300">
&nbsp;
<img src="https://github.com/TreB1eN/Pytorch0.4.1_Openpose/blob/master/data/hand_result.png" width="300">
</div>



## Train your model

This is a training procedure using COCO 2017 dataset.

### Download COCO 2017 dataset

```
cd data
bash getData.sh
```

If you already downloaded the dataset by yourself, please skip this procedure and change coco_dir in `entity.py` to the dataset path that was already downloaded.

### Setup COCO API

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../../
```

### Download [VGG-19 pretrained model](https://1drv.ms/u/s!AhMqVPD44cDOhx3dz655sCwOck2X) to `models` folder

### Generate and save image masks

Mask images are created in order to filter out people regions who were not labeled with any keypoints.
`vis` option can be used to visualize the mask generated from each image.

```
python gen_ignore_mask.py
```

### Train with COCO dataset

For each 1000 iterations, the recent weight parameters are saved as a weight file `model_iter_1000`.

```
python train.py
```

More configuration about training are in the `entity.py` file

## Related repository

- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).



## Citation

Please cite the original paper in your publications if it helps your research:    

```
@InProceedings{cao2017realtime,
  title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
  }
```