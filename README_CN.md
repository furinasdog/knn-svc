# kNN-SVC：基于加性合成与拼接平滑优化的鲁棒零样本歌声转换
[![许可证: MIT](https://img.shields.io/badge/许可证-MIT-yellow.svg)](LICENSE)

**欢迎提交 [Issues](https://github.com/SmoothKen/knn-svc/issues/new/choose) 反馈代码或演示中的问题**

本仓库提供 kNN-SVC 的推理代码。

## 环境配置

- 前置要求：Python 3.12、Conda
- 安装依赖：`pip install -r requirements.txt`
- 预训练模型可在 Releases 页面下载，下载后放入文件夹并通过命令行参数指定路径
- 支持以下三种转换方式，遇到问题欢迎通过 Issues 反馈

所有示例均假设输入为 16kHz 单声道音频。

## 1) 单文件 ➜ 单文件

运行主入口脚本，输出文件保存在源文件同级目录下，命名格式为：
`<源文件名>_to_<目标文件名>_knn_<ckpt_type>_<post_opt>.wav`

```bash
python scripts/inference.py /path/to/src.wav /path/to/style.wav \
    --ckpt_dir /path/to/ckpt_dir \
    --ckpt_type mix \
    --post_opt post_opt_0.2 \
    --topk 4 \
    --device cuda \
    --prioritize_f0 true \
    --tgt_loudness_db -16
```

参数说明：
- `--ckpt_type` 可选值：`mix`、`mix_harm_no_amp_*`、`mix_no_harm_no_amp_*`、`wavlm_only`、`wavlm_only_original`，其中 `harm` 表示加性合成条件
- `--post_opt` 平滑优化，可选 `no_post_opt` 或 `post_opt_0.2`

## 2) 数据集 ➜ 数据集

`src` 和 `tgt` 均为包含说话人子文件夹的数据集根目录。
转换后的音频将写入目标数据集父目录下自动创建的文件夹中，路径格式为：
`<tgt的父目录>/{src_name}_to_{tgt_name}_{ckpt_type}_post_opt_{post_opt}/`

```bash
python scripts/inference.py /path/to/src_dataset_root /path/to/tgt_dataset_root \
    --ckpt_dir /path/to/ckpt_dir \
    --ckpt_type mix \
    --post_opt post_opt_0.2 \
    --required_subset_file /path/to/split.csv
```

参数说明：
- `--required_subset_file` 可过滤需要处理的文件（CSV 格式）
- `--dur_limit` 限制目标池的总时长（单位为分钟，设为数字或留空表示不限制）

## 声码器微调

kNN 检索本身不需要训练（零样本），但如果预训练声码器在你的数据上效果不佳（如失真），可以用自己的数据微调 HiFi-GAN 声码器。

### 整体流程

```
准备音频 → 预处理特征 → 生成 CSV → 训练声码器 → 推理使用新权重
```

### 第一步：准备音频数据

将音频按说话人组织为子文件夹，要求 **16kHz 单声道** `.wav` 或 `.flac`：

```
data/
├── train/
│   ├── singer1/
│   │   ├── utt001.wav
│   │   └── utt002.wav
│   └── singer2/
│       └── utt003.wav
└── valid/
    └── singer3/
        └── utt004.wav
```

### 第二步：预处理特征

使用 WavLM 提取特征并计算 kNN 最近邻索引：

```bash
# 预处理训练集
python scripts/preprocess_train_data.py \
    --audio_path data/train \
    --out_path data/train_cached \
    --device cuda

# 预处理验证集
python scripts/preprocess_train_data.py \
    --audio_path data/valid \
    --out_path data/valid_cached \
    --device cuda
```

此脚本会自动：
- 用 WavLM-Large 提取每帧特征
- 构建 `pool.npy`（WavLM 特征池）和 `pool_harmonics.npy`（谐波特征池）
- 计算每条音频在说话人池中的 kNN 最近邻索引，保存为 `.pt` 文件
- 生成训练所需的 CSV 文件列表

### 第三步：训练声码器

```bash
python scripts/train_vocoder.py \
    --audio_root_path_train data/train \
    --audio_root_path_valid data/valid \
    --feature_root_path_train data/train_cached \
    --feature_root_path_valid data/valid_cached \
    --input_training_file data/train_cached_filelist.csv \
    --input_validation_file data/valid_cached_filelist.csv \
    --checkpoint_path output/ckpt \
    --training_epochs 1800
```

或直接调用底层模块：

```bash
python -m model.hifigan.ddsp_train \
    --audio_root_path_train data/train \
    --audio_root_path_valid data/valid \
    --feature_root_path_train data/train_cached \
    --feature_root_path_valid data/valid_cached \
    --input_training_file data/train_cached_filelist.csv \
    --input_validation_file data/valid_cached_filelist.csv \
    --checkpoint_path output/ckpt \
    --config model/hifigan/config_v1_wavlm.json \
    --training_epochs 1800 \
    --fine_tuning
```

训练完成后，checkpoint 保存在 `output/ckpt/` 目录下，命名格式为 `g_XXXXXXXX.pt`。

### 第四步：使用微调后的权重推理

将训练好的 checkpoint 放入一个目录，用 `--ckpt_dir` 指定：

```bash
python scripts/inference.py src.wav style.wav \
    --ckpt_dir output/ckpt \
    --ckpt_type mix_harm_no_amp \
    --post_opt post_opt_0.2 \
    --topk 4
```

### 推荐开源歌声数据集

| 数据集 | 语言 | 歌手数 | 时长 | 说明 |
|--------|------|--------|------|------|
| [OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io) | 中/英 | 40+ | ~40h | 多歌手，含专业与业余 |
| [M4Singer](https://github.com/M4Singer/M4Singer) | 中文 | 10 | ~6.5h | 业余歌手，带标注 |
| [Opencpop](https://wenet.org.cn/opencpop/) | 中文 | 1（女声） | ~100首 | NetEase 发布，高质量 |
| [PopCS](https://github.com/MoonInTheBowl/PopCS) | 中文 | 1 | ~100首 | 流行歌曲 |
| [CSD](https://www.kaggle.com/datasets/microsoft/children-song-dataset) | 韩语 | 2（童声） | ~1h | 儿童歌曲 |
| [Kising](https://drive.google.com/drive/folders/1746K1LkuJBgVdAqzSjHnqB4FZqOg0p0F) | 韩语 | 1 | ~2h | 女声独唱 |
| LibriSpeech | 英文 | 2000+ | 1000h | 语音（非歌声），原始训练用 |

> **提示**：对于中文歌声转换，推荐优先使用 OpenSinger 或 M4Singer 进行微调。数据量越大、音色越接近目标，效果越好。

### 关键配置参数

训练超参数在 `model/hifigan/config_v1_wavlm.json` 中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 16 | 多卡时按 GPU 数均分 |
| `learning_rate` | 0.0002 | 初始学习率 |
| `lr_decay` | 0.999 | 指数衰减系数 |
| `segment_size` | 7040 | 训练片段长度（采样点） |
| `n_harmonic` | 32 | 加性合成谐波数 |
| `with_harm` | true | 是否使用加性合成条件 |
| `sampling_rate` | 16000 | 采样率 |

## 状态与后续计划

我们计划统一 `ckpt_type` 的命名以减少混淆，但具体进度取决于后续研究进展。上述列出的现有选项将继续保持可用。

**相关链接**：

- Arxiv 论文：[https://arxiv.org/abs/2504.05686](https://arxiv.org/abs/2504.05686)
- 示例演示页面：[http://knnsvc.com/](http://knnsvc.com/)

![kNN-SVC 方法](./knn-svc.png)

**作者**：

- [Keren Shao](https://scholar.google.com/citations?user=jcQHdRgAAAAJ)
- [Ke Chen](https://www.knutchen.com/)
- [Matthew Baas](https://rf5.github.io/)
- [Shlomo Dubnov](http://dub.ucsd.edu/)

## 引用

```bibtex
@inproceedings{shao2025knn,
    title={kNN-SVC: Robust Zero-Shot Singing Voice Conversion with Additive Synthesis and Concatenation Smoothness Optimization},
    author={Shao, Keren and Chen, Ke and Baas, Matthew and Dubnov, Shlomo},
    booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={1--5},
    year={2025},
    organization={IEEE}
}
```
