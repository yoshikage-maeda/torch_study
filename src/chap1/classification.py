import json
from typing import Any
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from BaseTransform import BaseTransform
from ILSVRCPredictor import ILSVRCPredictor
def main():
    ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))

    predictor = ILSVRCPredictor(ILSVRC_class_index)

    # input
    image_file = './img/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file)

    # processing
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)
    inputs = img_transformed.unsqueeze_(0) # torch.size([1, 3, 224, 224])

    # input to model
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval() # inference mode
    out = net(inputs)
    result = predictor.predict_max(out)

    print(f'input img is: {result}')


if __name__ == '__main__':
    main()