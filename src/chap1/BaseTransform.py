from typing import Any
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
# 入力画像の前処理のクラス
class BaseTransform():
    """
    画像のサイズをリサイズし、色を標準化する。
    """

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize), # 短い辺の長さがresizeの大きさになる。
            transforms.CenterCrop(resize), # 画像中央をresize * resizeで切り取り
            transforms.ToTensor(), # Toransform to Torch Tensor
            transforms.Normalize(mean, std), # Normalize color information
        ])
    
    def __call__(self, img):
        return self.base_transform(img)

class ImageTransform():

    def __init__(self,resize, mean, std):
        self.data_transform = {
            # 訓練時にはデータオーグメンテーションを行う。
            'train':transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Toransform to Torch Tensor
            transforms.Normalize(mean, std), # Normalize color information
            ]),
            'val':transforms.Compose([
            transforms.Resize(resize), # 短い辺の長さがresizeの大きさになる。
            transforms.CenterCrop(resize), # 画像中央をresize * resizeで切り取り
            transforms.ToTensor(), # Toransform to Torch Tensor
            transforms.Normalize(mean, std), # Normalize color information
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
        pass
    
