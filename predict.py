import os
import sys
import yaml
import torch
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm
from macls.data_utils.featurizer import AudioFeaturizer
from macls.models import build_model
from macls.utils.utils import dict_to_object

def load_audio(audio_path):
    """加载音频文件"""
    waveform, sample_rate = sf.read(audio_path)
    # 确保音频是单声道
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    # 确保采样率是16kHz
    if sample_rate != 16000:
        raise ValueError(f'采样率必须是16kHz，当前采样率为{sample_rate}Hz')
    return waveform

def predict_audio(model, audio_featurizer, audio_path, device, class_labels):
    """预测单个音频文件"""
    try:
        # 加载音频
        waveform = load_audio(audio_path)
        waveform = torch.FloatTensor(waveform).unsqueeze(0)

        # 提取特征
        with torch.no_grad():
            feature = audio_featurizer(waveform)
            feature = feature.to(device)

            # 预测
            output = model(feature)
            output = torch.nn.functional.softmax(output, dim=-1)
            pred = output.argmax(dim=-1).item()
            score = output[0][pred].item()

        return class_labels[pred], score
    except Exception as e:
        return None, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预测音频分类')
    parser.add_argument('--configs', type=str, default='configs/train.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--audio_dir', type=str, default='dataset/audio/audio_data/test', help='要预测的音频文件夹路径')
    args = parser.parse_args()

    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    # 读取配置文件
    with open(args.configs, 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = dict_to_object(configs)

    # 获取分类标签
    with open(configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    class_labels = [l.replace('\n', '') for l in lines]

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建特征提取器
    audio_featurizer = AudioFeaturizer(
        feature_method=configs.preprocess_conf.feature_method,
        use_hf_model=configs.preprocess_conf.get('use_hf_model', False),
        method_args=configs.preprocess_conf.get('method_args', {})
    )

    # 创建模型
    model = build_model(
        input_size=audio_featurizer.feature_dim,
        configs=configs
    )
    model = model.to(device)

    # 加载模型权重
    if os.path.isdir(args.model_path):
        model_path = os.path.join(args.model_path, 'model.pth')
    else:
        model_path = args.model_path
    assert os.path.exists(model_path), f"{model_path}模型不存在！"
    checkpoint = torch.load(model_path, map_location=device)
    # 检查是否是完整的checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    # 移除module前缀
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 获取所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))

    # 预测所有音频文件
    results = []
    for audio_path in tqdm(audio_files, desc='预测进度'):
        pred_label, score = predict_audio(model, audio_featurizer, audio_path, device, class_labels)
        if pred_label is not None:
            results.append({
                'audio_path': audio_path,
                'pred_label': pred_label,
                'score': score
            })
            print(f'文件：{audio_path}')
            print(f'预测结果：{pred_label}')
            print(f'预测概率：{score:.4f}')
            print('-' * 50)

    # 统计结果
    print('\n预测统计：')
    label_counts = {}
    for result in results:
        label = result['pred_label']
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    for label, count in label_counts.items():
        print(f'{label}: {count}个文件')

if __name__ == '__main__':
    main() 