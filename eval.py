import os
import sys
import yaml
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import argparse
import functools
import time

from macls.trainer import MAClsTrainer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--configs', type=str, default='configs/train.yaml', help='配置文件路径')
    parser.add_argument('--resume_model', type=str, default='checkpoints/best_model.pth', help='模型路径')
    args = parser.parse_args()

    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    # 读取配置文件
    with open(args.configs, 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 创建训练器
    trainer = MAClsTrainer(configs=configs, use_gpu=True)

    # 评估模型
    loss, accuracy = trainer.evaluate(resume_model=args.resume_model)
    print(f'评估结果 - 损失: {loss:.5f}, 准确率: {accuracy:.5f}')

if __name__ == '__main__':
    main()
