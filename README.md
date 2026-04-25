# 🚗 **Improving Traffic Sign Detection in Adverse Snowy Environments via Frequency-Enhanced EfficientViT**

This repository contains the PyTorch implementation of the paper *"Improving Traffic Sign Detection under Adverse Snowy Environments via Frequency-Enhanced EfficientViT"* by **Yuhuan Wang**, **Tingting Wang**, **Yuan Wang**, **Tieji Zhang**, and **Yunfeng Hu**. In this work, we introduce the **Frequency Enhancement EfficientViT Network (FENet)**, designed to enhance detection robustness in snowy conditions and improve the accuracy of small traffic sign detection. Our paper has been accepted by **IEEE Transactions on Intelligent Transportation Systems (T-ITS)**. Full paper details will be updated once available online.

---

![Model Architecture](data/images/structure.png)

### 🔑 **Key Contributions:**

- A frequency compensation enhancement module (FCEM) based on gOctConv is designed to effectively model the complementary relationships among features at different frequencies and enhance their representation capability.

- A gaussian-weighted weather-adaptive loss (GW loss) is designed to overcome the limitations of traditional IoU loss in small object detection. By learning weather-related characteristics, GW loss adaptively enhances the detection and localization of small traffic signs under adverse weather conditions.

- Snow-CCTSDB is presented as a snowy traffic sign dataset created from CCTSDB using style transfer techniques to realistically simulate snowy traffic scenes. Extensive experiments on CCTSDB and Snow-CCTSDB demonstrate the effectiveness and robustness of our proposed FENet for traffic sign detection in snowy conditions.

---

## 🛠 **Installation**

Clone the repository and install the necessary dependencies:

```bash id="q0xx8y"
git clone https://github.com/your-username/FENet.git
cd FENet
pip install -r requirements.txt
```

---

## 📊 **Dataset & Weights**

![Dataset Example](data/images/dataset.png)

To evaluate FENet's robustness in adverse weather, we constructed Snow-CCTSDB, a custom dataset derived from the CCTSDB dataset. We used the Contrastive Unpaired Translation (CUT) framework to synthesize snowy conditions. The framework utilizes contrastive learning for unpaired image-to-image translation, simulating snow-induced degradations while preserving spatial annotations.

You can obtain the datasets through the following links:

* [CCTSDB Dataset](https://github.com/csust7zhangjm/CCTSDB.git)
* [Snow-CCTSDB Dataset](https://pan.baidu.com/s/1foMh8VqvJHyZrYr3lsm_JQ?pwd=y56r)

Please update the dataset paths in the configuration file:

```bash id="8hxr12"
data/SnowCCTSDB.yaml
```

The YAML file should look like this, specifying the paths to your datasets:

```yaml id="f2rt62"
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images
```

To reproduce the results, you'll need the pre-trained weights for the network and EfficientViT. Download them from the links below, or use YOLOv5's pre-trained weights:

* [Network Weights](https://pan.baidu.com/s/1TMiBeoXuBZXp0pskEPLq4Q?pwd=sd84)
* [EfficientViT Weights](https://pan.baidu.com/s/10y2BqWImIcR201PlVdjD0Q?pwd=5i4r)

Place the downloaded weights as follows:

```text id="nueyz1"
pretrained_pth/pretrained_weight.pt
checkpoints/efficientViT/b2-r288.pt
```

---

## 🚀 **Training**

To train the FENet model, run the following command:

```bash id="t38sc6"
python train_FENet.py \
  --data data/SnowCCTSDB.yaml \
  --cfg models/FENet.yaml \
  --weights pretrained_pth/pretrained_weight.pt \
  --imgsz 640 \
  --batch-size 8 \
  --epochs 100 \
  --device 0
```

---

## ✅ **Validation**

To validate the trained model, use this command:

```bash id="new1lb"
python val_FENet.py \
  --data data/SnowCCTSDB.yaml \
  --weights runs/train/<exp_name>/weights/best.pt \
  --imgsz 640 \
  --batch-size 32 \
  --device 0
```

---

## 📑 **Citation**

If you find this project useful, please consider citing our paper. Your support is greatly appreciated!

---
