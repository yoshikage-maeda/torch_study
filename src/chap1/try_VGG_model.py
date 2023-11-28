from typing import Any
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from BaseTransform import BaseTransform
def main():
    # create instances vgg-16 model
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval() # inference mode

    print(net) # print netork info

    # image_path
    path = './img/goldenretriever-3724972_640.jpg'
    img = Image.open(path) # height, width, colorRGB

    # processing images and display processed images
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)

    # (color, height, width) -> (height, width, color) and limited [0,1]
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.savefig('./img/transform.png')
    plt.close()
    plt.imshow(img)
    plt.savefig('./img/original_img.png')
if __name__=='__main__':
    main()