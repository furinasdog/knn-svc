"""
声码器（HiFi-GAN）微调训练启动脚本

用法：
    python scripts/train_vocoder.py \
        --audio_root_path_train data/train_audio \
        --audio_root_path_valid data/valid_audio \
        --feature_root_path_train data/train_cached \
        --feature_root_path_valid data/valid_cached \
        --input_training_file data/train_filelist.csv \
        --input_validation_file data/valid_filelist.csv \
        --checkpoint_path output_ckpt \
        --config model/hifigan/config_v1_wavlm.json \
        --training_epochs 1800

也可通过 shell 直接调用底层训练模块：
    python -m model.hifigan.ddsp_train [args...]
"""

import argparse
import os
import subprocess
import sys


def main():
	parser = argparse.ArgumentParser(
		description='kNN-SVC 声码器微调训练',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
示例:
  # 完整训练流程
  python scripts/train_vocoder.py \\
      --audio_root_path_train data/OpenSinger/train \\
      --audio_root_path_valid data/OpenSinger/valid \\
      --feature_root_path_train data/OpenSinger/train_cached \\
      --feature_root_path_valid data/OpenSinger/valid_cached \\
      --input_training_file data/OpenSinger/train_cached_filelist.csv \\
      --input_validation_file data/OpenSinger/valid_cached_filelist.csv \\
      --checkpoint_path output/OpenSinger_ckpt \\
      --training_epochs 1800

  # 从已有 checkpoint 继续训练（修改 ddsp_train.py 中注释掉 cp_g = None）
  # 默认每次从头开始训练，如需断点续训请编辑 model/hifigan/ddsp_train.py
		"""
	)

	parser.add_argument('--audio_root_path_train', type=str, required=True)
	parser.add_argument('--audio_root_path_valid', type=str, required=True)
	parser.add_argument('--feature_root_path_train', type=str, required=True)
	parser.add_argument('--feature_root_path_valid', type=str, required=True)
	parser.add_argument('--input_training_file', type=str, required=True)
	parser.add_argument('--input_validation_file', type=str, required=True)
	parser.add_argument('--checkpoint_path', type=str, default='output_ckpt')
	parser.add_argument('--config', type=str, default='model/hifigan/config_v1_wavlm.json')
	parser.add_argument('--training_epochs', type=int, default=1800)
	parser.add_argument('--stdout_interval', type=int, default=25)
	parser.add_argument('--validation_interval', type=int, default=1000)
	parser.add_argument('--fp16', type=bool, default=False)
	parser.add_argument('--fine_tuning', action='store_true', default=True,
						help='启用微调模式（加载预计算特征）')

	args = parser.parse_args()

	cmd = [
		sys.executable, '-m', 'model.hifigan.ddsp_train',
		'--audio_root_path_train', args.audio_root_path_train,
		'--audio_root_path_valid', args.audio_root_path_valid,
		'--feature_root_path_train', args.feature_root_path_train,
		'--feature_root_path_valid', args.feature_root_path_valid,
		'--input_training_file', args.input_training_file,
		'--input_validation_file', args.input_validation_file,
		'--checkpoint_path', args.checkpoint_path,
		'--config', args.config,
		'--training_epochs', str(args.training_epochs),
		'--stdout_interval', str(args.stdout_interval),
		'--validation_interval', str(args.validation_interval),
	]

	if args.fine_tuning:
		cmd.append('--fine_tuning')

	print("[INFO] 启动训练命令:")
	print(' '.join(cmd))
	print()

	result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
	sys.exit(result.returncode)


if __name__ == '__main__':
	main()
