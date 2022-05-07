## Learning Clustering for Motion Segmentation
Created by <a href="http://xu-xun.com" target="_blank">Xun Xu</a>.

![prediction example](./doc/teaser.png)

### Introduction
This is the official repository for <a href='https://arxiv.org/pdf/1904.02075.pdf'> Learning Clustering for Motion Segmentation [1]</a> . We proposed a deep neural network based clustering framework to learned from labeled clustering dataset. Experiments are carried out on both rigid 3D motio segmentation and more challenging non-rigid motion segmentation benchmarks.

You can also replace the backbone network with <a href="https://github.com/charlesq34/pointnet"> PointNet </a>, <a href="https://github.com/charlesq34/pointnet2"> PointNet++ </a>or other point cloud backbones.

We currently released tensorflow code for experiments on KT3DMoSeg dataset [2, 3]. If you can implement this algorithm in PyTorch I am very happy to provide the link to your repository. You are welcome to report any bugs you would identify. Should you have any concerns or experience any issues please raise in Issues so that all people can benefit from the discussions.

### Installation
This code has been tested on Pyhon3.6, TensorFlow1.14, CUDA 10.0, cuDNN 7.0 and Ubuntu 18.04

### Dataset
You should first download the data for KT3MoSeg from: https://www.dropbox.com/s/h6ub9pt9dk4j3h8/KT3DMoSegClips.zip?dl=0 . Then put the unzipped files under ./Dataset . The ./KT3DMoSeg/Seq contains the original clips and the ./KT3DMoSeg/Data contains all processed trajectories.

### Usage
You can run the cross-validation experiment by running the following script. You only need to specify the GPU you want to use for training and keep all other hyperparameters unchanged.

python  CrossValid_KT3DMoSeg.py --GPU #GPU

### Reference
[1] Xun Xu, Loong-Fah Cheong, Zhuwen Li, Le Zhang and Ce Zhu. "Learning Clustering for Motion Segmentation." IEEE Transactions on Circuits and Systems for Video Technology (2021).

[2] Xun Xu, Loong-Fah Cheong, and Zhuwen Li. "Motion segmentation by exploiting complementary geometric models." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[3] Xun Xu, Loong Fah Cheong, and Zhuwen Li. "3D Rigid Motion Segmentation with Mixed and Unknown Number of Models." IEEE Transactions on Pattern Analysis and Machine Intelligence (2019).
