"""
训练数据预处理脚本

功能：
1. 用 WavLM 提取音频特征
2. 构建 pool.npy / pool_harmonics.npy
3. 计算每条音频的 kNN 最近邻索引
4. 生成训练所需的 CSV 文件列表

用法：
    python scripts/preprocess_train_data.py \
        --audio_path /path/to/train_audio \
        --out_path /path/to/cached_features \
        --device cuda

目录结构要求：
    audio_path/
    ├── speaker1/
    │   ├── utt001.wav
    │   └── utt002.wav
    └── speaker2/
        └── utt003.wav

输出结构：
    out_path/
    ├── speaker1/
    │   ├── pool.npy              # 该说话人所有帧的 WavLM 特征
    │   ├── pool_harmonics.npy    # 谐波振幅特征
    │   ├── pool_f0.npy           # F0 特征
    │   ├── pool_spec.npy         # 频谱特征
    │   ├── utt001.pt             # 每条音频的 kNN 索引 (pickle)
    │   └── utt002.pt
    └── speaker2/
        └── ...
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from hubconf import wavlm_large
from model.dataset import per_spk_extract


def generate_csv_filelist(audio_root: str, feat_root: str, output_csv: str, extensions=None):
	"""根据预处理后的目录结构生成训练用 CSV 文件列表。"""
	if extensions is None:
		extensions = {'.wav', '.flac'}

	audio_root = Path(audio_root)
	feat_root = Path(feat_root)

	rows = []
	for audio_file in sorted(audio_root.rglob('*')):
		if audio_file.suffix.lower() not in extensions:
			continue

		rel_path = audio_file.relative_to(audio_root)
		feat_path = rel_path.with_suffix('.pt')

		full_feat_path = feat_root / feat_path
		if not full_feat_path.exists():
			print(f"[WARN] 特征文件不存在，跳过: {full_feat_path}")
			continue

		rows.append(f"{rel_path.as_posix()},{feat_path.as_posix()}")

	with open(output_csv, 'w', encoding='utf-8') as f:
		f.write("audio_path,feat_path\n")
		for row in rows:
			f.write(row + "\n")

	print(f"[INFO] 已生成 CSV: {output_csv} ({len(rows)} 条记录)")


def main():
	parser = argparse.ArgumentParser(
		description='kNN-SVC 声码器微调 — 训练数据预处理',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
示例:
  # 预处理训练集
  python scripts/preprocess_train_data.py \\
      --audio_path data/OpenSinger/train \\
      --out_path data/OpenSinger/train_cached \\
      --device cuda

  # 预处理验证集
  python scripts/preprocess_train_data.py \\
      --audio_path data/OpenSinger/valid \\
      --out_path data/OpenSinger/valid_cached \\
      --device cuda

  # 仅生成 CSV（已有预处理结果时）
  python scripts/preprocess_train_data.py \\
      --audio_path data/OpenSinger/train \\
      --out_path data/OpenSinger/train_cached \\
      --csv_only \\
      --train_csv data/train.csv
		"""
	)

	parser.add_argument('--audio_path', type=str, required=True,
						help='音频数据根目录（包含说话人子文件夹）')
	parser.add_argument('--out_path', type=str, required=True,
						help='特征缓存输出目录')
	parser.add_argument('--device', type=str, default='cuda',
						help='设备 (cuda / cpu)')
	parser.add_argument('--matching_layer', type=int, default=6,
						help='WavLM 匹配层索引 (默认 6)')
	parser.add_argument('--synthesis_layer', type=int, default=6,
						help='WavLM 合成层索引 (默认 6)')
	parser.add_argument('--topk', type=int, default=4,
						help='kNN 的 k 值')
	parser.add_argument('--seed', type=int, default=123,
						help='随机种子')

	parser.add_argument('--train_csv', type=str, default=None,
						help='输出训练集 CSV 路径')
	parser.add_argument('--csv_only', action='store_true',
						help='仅生成 CSV，不执行特征提取')

	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if not args.csv_only:
		print(f"[INFO] 加载 WavLM-Large ...")
		wavlm = wavlm_large(pretrained=True, progress=True, device=args.device)

		synth_weights = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=25).float().to(args.device)[:, None]
		match_weights = F.one_hot(torch.tensor(args.matching_layer), num_classes=25).float().to(args.device)[:, None]

		print(f"[INFO] 匹配层权重: 第 {args.matching_layer} 层")
		print(f"[INFO] 合成层权重: 第 {args.synthesis_layer} 层")
		print(f"[INFO] 音频目录: {args.audio_path}")
		print(f"[INFO] 输出目录: {args.out_path}")

		per_spk_extract(
			wavlm=wavlm,
			device=args.device,
			ls_path=Path(args.audio_path),
			out_path=Path(args.out_path),
			synth_weights=synth_weights,
			match_weights=match_weights,
			save_pool_only=False,
		)
		print("[INFO] 特征提取完成!")

	if args.train_csv:
		generate_csv_filelist(args.audio_path, args.out_path, args.train_csv)
	elif not args.csv_only:
		default_csv = str(Path(args.out_path).parent / (Path(args.out_path).name + '_filelist.csv'))
		generate_csv_filelist(args.audio_path, args.out_path, default_csv)


if __name__ == '__main__':
	main()
