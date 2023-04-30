from .ConvNet import ConvNet
from .PretrainedConvNet import PretrainedConvNet
from .VisionTransformer import ViT

MODELS = {
    "ConvNet": ConvNet,
    "PretrainedConvNet": PretrainedConvNet,
    "ViT": ViT
}