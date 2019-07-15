# Context-Aware Visual Policy Network for Sequence-Level Image Captioning

This repository contains the code for the following papers:

- Daqing Liu, Zheng-Jun Zha, Hanwang Zhang, Yongdong Zhang, Feng Wu, *Context-Aware Visual Policy Network for Sequence-Level Image Captioning*. in ACM MM, 2018. ([PDF]((https://arxiv.org/abs/1808.05864)))

- Zheng-Jun Zha, Daqing Liu, Hanwang Zhang, Yongdong Zhang, Feng Wu, *Context-Aware Visual Policy Network for Fine-Grained Image Captioning*. in TPAMI, 2019. (Extended journal version. [PDF]((https://arxiv.org/abs/1906.02365)))

## Installation
1. Install Python 3 ([Anaconda](https://www.anaconda.com/distribution/) recommended).
2. Install [Pytorch](https://pytorch.org/) v1.0 or higher:
```
pip3 install torch torchvision
```
3. Install Java JDK (for METEOR Metric):
```
apt install default-jdk
```
4. Clone with Git, and then enter the root directory:
```
git clone --recursive https://github.com/daqingliu/CAVP.git && cd CAVP
```

## Download Data
1. Download the image features (download link coming soon) extracted from [bottom-up-attention](https://github.com/daqingliu/bottom-up-attention) into ```data``` and unzip it.
2. Download coco annotations ([h5](https://drive.google.com/open?id=1XzKig7BvPISCb818_qMVIjyB_Al3Afov) and [json](https://drive.google.com/open?id=1QJ4VtgzrKMXdRUQ5wjWe0tZiXO0-5sNj)) into ```data```.

## Training and Test
Just simply run:
```
bash run_train.sh
bash run_eval.sh
```

## Citation
```
@article{zha2019context,
  title={Context-aware visual policy network for fine-grained image captioning},
  author={Zha, Zheng-Jun and Liu, Daqing and Zhang, Hanwang and Zhang, Yongdong and Wu, Feng},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
}
```

## Acknowledgements
Part of this repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
