import importlib

from loguru import logger
from .campplus import CAMPPlus
from .ecapa_tdnn import EcapaTdnn
from .eres2net import ERes2Net, ERes2NetV2
from .panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from .res2net import Res2Net
from .resnet_se import ResNetSE
from .tdnn import TDNN

__all__ = ['EcapaTdnn']


def build_model(input_size, configs):
    """构建模型

    :param input_size: 输入特征大小
    :param configs: 模型配置
    :return: 模型
    """
    model_name = configs.model_conf.use_model
    model_args = configs.model_conf.model_args.copy()
    # 移除input_size参数，避免重复传递
    if 'input_size' in model_args:
        del model_args['input_size']
    model = eval(model_name)(input_size=input_size, **model_args)
    logger.info(f'成功创建模型：{model_name}，参数为：{model_args}')
    return model
