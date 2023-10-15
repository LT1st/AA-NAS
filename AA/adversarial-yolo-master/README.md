# 流程
## 训练patch
## patch贴上去
## 验证精度
## loss with bkg(without human) and path




# Adversarial YOLO
This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2

This work corresponds to the following paper: https://arxiv.org/abs/1904.08653:
```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

If you use this work, please cite this paper.

# What you need
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed, to do this see: https://pytorch.org/
To visualise progress we use tensorboardX which can be installed using pip:
```
pip install tensorboardX tensorboard
pip install tensorboardX tensorboard -i http://pypi.douban.com/simple Flask  -- trusted-host pypi.douban.com
或者
pip install tensorboardX -i http://pypi.douban.com/simple -- trusted-host pypi.douban.com

```
No installation is necessary, you can simply run the python code straight from this directory.

Make sure you have the YOLOv2 MS COCO weights:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```

Get the INRIA dataset:
```
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
tar xf inria.tar
mv INRIAPerson inria
cp -r yolo-labels inria/Train/pos/
```

# Generating a patch
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

You can generate this patch by running:
```
python train_patch.py paper_obj
```

# Exp
About 11s for every inter in 3090 24G in Linux. 
About 35s for every inter in 3090 24G in Win. 

# Architecture of Files 
```
├─cfg                           patch参数、网络参数
├─data                          数据处理脚本
│  └─INRIAPerson
├─DGXContainer
├─inria                         数据集
│  ├─70X134H96
│  │  └─Test
│  │      └─pos
│  ├─96X160H96
│  │  └─Train
│  │      └─pos
│  ├─Test                           测试集，需要跑脚本生成yolo-labels
│  │  ├─annotations
│  │  ├─neg
│  │  └─pos
│  │      └─yolo-labels
│  ├─test_64x128_H96
│  │  └─pos
│  ├─Train                          训练集，需要跑脚本生成yolo-labels
│  │  ├─annotations
│  │  ├─neg
│  │  └─pos
│  │      └─yolo-labels
│  └─train_64x128_H96
│      └─pos
├─layers                        网络结构
│  └─batchnorm
│      ├─bn_lib
│      └─src
├─lutao_exp                     实验部分记录
│  ├─ex1
│  ├─ex2
│  └─ex3
├─models                        模型
│  └─__pycache__
├─non_printability
├─oude versies
│  └─lab
├─tools                         
│  └─lmdb
├─weights                       模型权重
├─yolo-labels                   用于train的label
└─__pycache__
```


# details
```
ep_det_loss += det_loss.detach().cpu().numpy()
ep_nps_loss += nps_loss.detach().cpu().numpy()
ep_tv_loss += tv_loss.detach().cpu().numpy()
ep_loss += loss


nps_loss = nps*0.01
tv_loss = tv*2.5
```
# LOG
## 第一阶段： 复现指标
6.6结束

## 第二阶段：  开始实验
### 测试A4纸
- 发现了对于亮度信息没有适配，没有对于纸张的适配
- 没有边框
- 灰度亮度均衡化？

# 变量
max_lab 最多检测人数