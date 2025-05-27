import os
import numpy as np
import librosa
import parselmouth
import webrtcvad
import torch
import torchaudio
from tqdm import tqdm
from loguru import logger

def extract_volume_features(audio, sr):
    """提取音量相关特征"""
    # 计算音量
    volume = np.abs(audio)
    # 平均音量
    mean_volume = np.mean(volume)
    # 音量变化范围
    volume_range = np.ptp(volume)
    # 音量标准差
    volume_std = np.std(volume)
    
    return {
        'mean_volume': mean_volume,
        'volume_range': volume_range,
        'volume_std': volume_std
    }

def extract_pitch_features(audio, sr):
    """提取音调相关特征"""
    try:
        # 使用parselmouth提取基频
        sound = parselmouth.Sound(audio, sr)
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        
        # 检查f0数组是否为空
        if len(f0) == 0:
            return {
                'mean_f0': 0.0,
                'f0_range': 0.0,
                'f0_std': 0.0,
                'f0_change_rate': 0.0,
                'f0_stability': 0.0
            }
        
        # 将0值替换为nan
        f0[f0 == 0] = np.nan
        
        # 检查是否有有效的f0值
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) == 0:
            return {
                'mean_f0': 0.0,
                'f0_range': 0.0,
                'f0_std': 0.0,
                'f0_change_rate': 0.0,
                'f0_stability': 0.0
            }
        
        # 基频统计特征
        mean_f0 = np.mean(valid_f0)  # 使用有效值计算
        f0_range = np.ptp(valid_f0)  # 使用有效值计算
        f0_std = np.std(valid_f0)    # 使用有效值计算
        
        # 音调变化率
        f0_diff = np.diff(valid_f0)  # 使用有效值计算
        if len(f0_diff) > 0:
            f0_change_rate = np.mean(np.abs(f0_diff))
        else:
            f0_change_rate = 0.0
        
        # 音调稳定性 (使用变异系数)
        f0_stability = f0_std / mean_f0 if mean_f0 != 0 else 0.0
        
        # 确保所有值都是有限的
        features = {
            'mean_f0': float(mean_f0),
            'f0_range': float(f0_range),
            'f0_std': float(f0_std),
            'f0_change_rate': float(f0_change_rate),
            'f0_stability': float(f0_stability)
        }
        
        # 检查并替换任何非有限值
        for key in features:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"提取音调特征时出错: {str(e)}")
        return {
            'mean_f0': 0.0,
            'f0_range': 0.0,
            'f0_std': 0.0,
            'f0_change_rate': 0.0,
            'f0_stability': 0.0
        }

def extract_speech_rate_features(audio, sr):
    """提取语速相关特征"""
    try:
        # 确保音频数据是16位整数格式
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # 使用webrtcvad进行语音活动检测
        vad = webrtcvad.Vad(3)  # 设置激进程度为3
        frame_duration = 30  # 30ms
        frame_size = int(sr * frame_duration / 1000)
        
        # 确保音频长度足够
        if len(audio) < frame_size:
            return {
                'vad_ratio': 0,
                'mean_segment_length': 0,
                'segment_length_std': 0
            }
        
        # 将音频分成帧
        frames = []
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) == frame_size:
                frames.append(frame)
        
        if not frames:
            return {
                'vad_ratio': 0,
                'mean_segment_length': 0,
                'segment_length_std': 0
            }
        
        # 检测每一帧是否为语音
        is_speech = []
        for frame in frames:
            try:
                is_speech.append(vad.is_speech(frame.tobytes(), sr))
            except Exception as e:
                logger.warning(f"处理帧时出错: {str(e)}")
                is_speech.append(False)
        
        vad_ratio = sum(is_speech) / len(is_speech) if is_speech else 0
        
        # 计算语音片段长度
        speech_segments = []
        current_segment = 0
        for is_speech_frame in is_speech:
            if is_speech_frame:
                current_segment += 1
            elif current_segment > 0:
                speech_segments.append(current_segment)
                current_segment = 0
        if current_segment > 0:
            speech_segments.append(current_segment)
        
        # 语音片段统计
        if speech_segments:
            mean_segment_length = np.mean(speech_segments)
            segment_length_std = np.std(speech_segments)
        else:
            mean_segment_length = 0
            segment_length_std = 0
        
        return {
            'vad_ratio': float(vad_ratio),  # 确保返回Python原生类型
            'mean_segment_length': float(mean_segment_length),
            'segment_length_std': float(segment_length_std)
        }
    except Exception as e:
        logger.error(f"提取语音速率特征时出错: {str(e)}")
        return {
            'vad_ratio': 0,
            'mean_segment_length': 0,
            'segment_length_std': 0
        }

def extract_quality_features(audio, sr):
    """提取音质相关特征"""
    try:
        # 计算频谱
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D))
        
        # 信噪比 (SNR)
        signal_mask = S_db > np.median(S_db)
        noise_mask = ~signal_mask
        
        if np.any(signal_mask) and np.any(noise_mask):
            signal_power = np.mean(S_db[signal_mask])
            noise_power = np.mean(S_db[noise_mask])
            snr = signal_power - noise_power if noise_power != 0 else 0.0
        else:
            snr = 0.0
        
        # 谐波噪声比 (HNR)
        try:
            sound = parselmouth.Sound(audio, sr)
            hnr = sound.to_harmonicity()
            mean_hnr = np.mean(hnr.values)
            if np.isnan(mean_hnr):
                mean_hnr = 0.0
        except Exception as e:
            logger.warning(f"计算HNR时出错: {str(e)}")
            mean_hnr = 0.0
        
        # 频谱质心
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_spectral_centroid = np.mean(spectral_centroid)
            if np.isnan(mean_spectral_centroid):
                mean_spectral_centroid = 0.0
        except Exception as e:
            logger.warning(f"计算频谱质心时出错: {str(e)}")
            mean_spectral_centroid = 0.0
        
        return {
            'snr': float(snr),
            'hnr': float(mean_hnr),
            'spectral_centroid': float(mean_spectral_centroid)
        }
    except Exception as e:
        logger.error(f"提取音质特征时出错: {str(e)}")
        return {
            'snr': 0.0,
            'hnr': 0.0,
            'spectral_centroid': 0.0
        }

def extract_all_features(audio_path, sr=16000):
    """提取所有特征"""
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # 提取各类特征
    volume_features = extract_volume_features(audio, sr)
    pitch_features = extract_pitch_features(audio, sr)
    speech_rate_features = extract_speech_rate_features(audio, sr)
    quality_features = extract_quality_features(audio, sr)
    
    # 合并所有特征
    all_features = {
        **volume_features,
        **pitch_features,
        **speech_rate_features,
        **quality_features
    }
    
    return all_features

def process_audio_files(data_list_path, save_dir, max_duration=100):
    """处理音频文件列表，提取并保存特征"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取数据列表
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建新的特征列表文件
    new_list_path = data_list_path.replace('.txt', '_allfeat.txt')
    with open(new_list_path, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, desc=f"处理{data_list_path}"):
            audio_path, label = line.strip().split('\t')
            
            try:
                # 提取特征
                features = extract_all_features(audio_path)
                
                # 将所有特征转换为标量值
                feature_dict = {}
                for key, value in features.items():
                    # 处理numpy数值类型
                    if isinstance(value, (np.number, int, float)):
                        feature_dict[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        if value.ndim == 0:  # 标量数组
                            feature_dict[key] = float(value)
                        elif value.ndim == 1:  # 一维数组
                            feature_dict[key] = value.tolist()
                        else:
                            logger.warning(f"跳过多维数组特征 {key}")
                    else:
                        logger.warning(f"跳过不支持的特征类型 {key}: {type(value)}")
                
                # 保存特征
                feature_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.npz")
                np.savez(feature_path, **feature_dict)
                
                # 写入新的列表文件
                f.write(f"{audio_path}\t{feature_path}\t{label}\n")
                
            except Exception as e:
                logger.error(f"处理文件 {audio_path} 时出错: {str(e)}")
                continue
    
    logger.info(f"特征提取完成，新列表文件保存在: {new_list_path}")

def main():
    # 处理训练集
    process_audio_files(
        data_list_path='dataset/train_list.txt',
        save_dir='dataset/features/train'
    )
    
    # 处理测试集
    process_audio_files(
        data_list_path='dataset/test_list.txt',
        save_dir='dataset/features/test'
    )

if __name__ == '__main__':
    main() 