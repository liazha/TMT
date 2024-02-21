# TMT

A novel Token-disentangling Mutual Transformer(TMT) for multimodal emotion recognition. The TMT can effectively disentangle inter-modality emotion consistency features and intra-modality
emotion heterogeneity features and mutually fuse them for more comprehensive multimodal emotion representations by introducing two primary modules, namely multimodal emotion Token disentanglement and Token mutual Transformer. 

### Features

- Train, test and compare in a unified framework.
- Supports Our TMT model.
- Supports 3 MSA datasets: [MOSI](https://ieeexplore.ieee.org/abstract/document/7742221), [MOSEI](https://aclanthology.org/P18-1208.pdf), and [CH-SIMS](https://aclanthology.org/2020.acl-main.343/).
- Easy to use, provides Python APIs and commandline tools.
- Experiment with fully customized multimodal features extracted by [MMSA-FET](https://github.com/thuiar/MMSA-FET) toolkit.

## 1. Get Started

> **Note:** From version 2.0, we packaged the project and uploaded it to PyPI in the hope of making it easier to use. If you don't like the new structure, you can always switch back to `v_1.0` branch. 

### 1.1 Use Python API

- Import and use in any python file:

  ```python
  python test.py
  python test5.py

- For more detailed usage, please refer to [APIs](https://github.com/thuiar/MMSA/wiki/APIs).



### 1.3 Clone & Edit the Code

- Clone this repo and install requirements.
  ```bash
  $ git clone https://github.com/cug-ygh/TMT
  ```


## 2. Datasets

MMSA currently supports MOSI, MOSEI, and CH-SIMS dataset. Use the following links to download raw videos, feature files and label files. You don't need to download raw videos if you're not planning to run end-to-end tasks. 

- [BaiduYun Disk](https://pan.baidu.com/s/1XmobKHUqnXciAm7hfnj2gg) `code: mfet`
- [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing)

SHA-256 for feature files:

```text
`MOSI/Processed/unaligned_50.pkl`:  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`
`MOSI/Processed/aligned_50.pkl`:    `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`
`MOSEI/Processed/unaligned_50.pkl`: `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`
`MOSEI/Processed/aligned_50.pkl`:   `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`
`SIMS/Processed/unaligned_39.pkl`:  `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`
```

Our uses feature files that are organized as follows:

```python
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```
## 5. Citation

- [CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality](https://www.aclweb.org/anthology/2020.acl-main.343/)
- [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis](https://arxiv.org/abs/2102.04830)
- [M-SENA: An Integrated Platform for Multimodal Sentiment Analysis]()

Please cite our paper if you find our work useful for your research:

```
@inproceedings{yu2020ch,
  title={CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality},
  author={Yu, Wenmeng and Xu, Hua and Meng, Fanyang and Zhu, Yilin and Ma, Yixiao and Wu, Jiele and Zou, Jiyun and Yang, Kaicheng},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3718--3727},
  year={2020}
}
```

```
@inproceedings{yu2021learning,
  title={Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis},
  author={Yu, Wenmeng and Xu, Hua and Yuan, Ziqi and Wu, Jiele},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={10790--10797},
  year={2021}
}
```

