import os
import sys
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import AudioDataset
from models.ecapatdnn import EcapaTdnn
from loguru import logger
import numpy as np
from tqdm import tqdm
from tools.extract_all_features import process_audio_files
import yaml

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="训练中"):
        audio_features = batch['audio_features'].to(device)
        extra_features = batch['extra_features'].to(device) if batch['extra_features'] is not None else None
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(audio_features, extra_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            audio_features = batch['audio_features'].to(device)
            extra_features = batch['extra_features'].to(device) if batch['extra_features'] is not None else None
            labels = batch['label'].to(device)
            
            outputs = model(audio_features, extra_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def get_num_classes(label_list_path):
    """获取类别数量"""
    with open(label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len([l.strip() for l in lines])

def check_and_extract_features():
    """检查并提取特征"""
    if not os.path.exists('dataset/train_list_allfeat.txt') or not os.path.exists('dataset/test_list_allfeat.txt'):
        logger.info("开始提取特征...")
        # 处理训练集
        logger.info("处理训练集...")
        process_audio_files(
            data_list_path='dataset/train_list.txt',
            save_dir='dataset/features/train'
        )
        # 处理测试集
        logger.info("处理测试集...")
        process_audio_files(
            data_list_path='dataset/test_list.txt',
            save_dir='dataset/features/test'
        )
        logger.info("特征提取完成！")
    else:
        logger.info("特征文件已存在，跳过特征提取步骤。")

def main():
    # 加载配置
    config = load_config('configs/train.yaml')
    dataset_conf = config['dataset_conf']
    train_conf = config['train_conf']
    preprocess_conf = config['preprocess_conf']
    model_conf = config['model_conf']
    
    # 设置默认的log_interval
    if 'log_interval' not in train_conf:
        train_conf['log_interval'] = 1
        logger.info("使用默认的log_interval: 1")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查并提取特征
    check_and_extract_features()
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 获取类别数量
    num_classes = get_num_classes(dataset_conf['label_list_path'])
    logger.info(f"类别数量: {num_classes}")
    
    # 加载数据集
    train_dataset = AudioDataset('dataset/train_list_allfeat.txt')
    val_dataset = AudioDataset('dataset/test_list_allfeat.txt')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataset_conf['dataLoader']['batch_size'], 
        shuffle=True, 
        num_workers=dataset_conf['dataLoader']['num_workers'],
        drop_last=dataset_conf['dataLoader']['drop_last']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataset_conf['eval_conf']['batch_size'], 
        shuffle=False, 
        num_workers=dataset_conf['dataLoader']['num_workers']
    )
    
    # 创建模型
    model = EcapaTdnn(
        input_size=preprocess_conf['method_args']['num_mel_bins'],
        hidden_size=model_conf['model_args']['hidden_size'],
        output_size=model_conf['model_args']['output_size'],
        num_class=num_classes,
        use_se=model_conf['model_args']['use_se'],
        se_reduction=model_conf['model_args']['se_reduction'],
        use_attention=model_conf['model_args']['use_attention'],
        num_heads=model_conf['model_args']['num_heads'],
        use_residual=model_conf['model_args']['use_residual'],
        use_layer_norm=model_conf['model_args']['use_layer_norm'],
        use_dropout=model_conf['model_args']['use_dropout'],
        dropout_rate=model_conf['model_args']['dropout_rate'],
        use_batch_norm=model_conf['model_args']['use_batch_norm'],
        use_group_norm=model_conf['model_args']['use_group_norm'],
        num_groups=model_conf['model_args']['num_groups']
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=train_conf['label_smoothing'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_conf['learning_rate'],
        weight_decay=train_conf['optimizer_args']['weight_decay'],
        betas=(train_conf['optimizer_args']['beta1'], train_conf['optimizer_args']['beta2'])
    )
    
    # 训练参数
    num_epochs = train_conf['max_epoch']
    best_val_acc = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/best_model.pth')
            logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 定期保存检查点
        if (epoch + 1) % train_conf['log_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
