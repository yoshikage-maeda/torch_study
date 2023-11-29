from pathlib import Path
import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
from Dataset import HymenopteraDataset
from BaseTransform import ImageTransform
from torchvision import models
from tqdm import tqdm
HOME_DIR = Path(os.getcwd())
DATA_DIR = HOME_DIR / 'data' / 'hymenoptera_data'

def make_datapath_list(phase='train'):
    target_path = DATA_DIR / f'{phase}' / '**' / '*.jpg'

    pathlist = []

    for path in glob.glob(str(target_path)):
        pathlist.append(path)
    
    return pathlist

def train_model(net, data_loders_dict, criterion, optimizer, numepochs):

    for epoch in range(numepochs):
        print(f'Epoch {epoch+1}/{numepochs}')
        print('--------------------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0.0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(data_loders_dict[phase]):

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = epoch_loss / len(data_loders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(data_loders_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss}, Acc:{epoch_acc}')
def main():
    BATCH_SIZE = 32
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # filepath
    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')
    train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # dict
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}

    #model
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)

    # モデルの最後の出力ユニットをアリとハチのふたつに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.train()

    # loss function
    critetion = nn.CrossEntropyLoss()

    # 転移学習では学習させるパラメータを、変数params_to_updateに格納する
    params_to_update = []

    update_param_names = ['classifier.6.weight', 'classifier.6.bias']

    # 学習するパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False

    # params_to_update の中身を確認
    print(params_to_update)

    # setting opt
    optimizer = optim.SGD(params=params_to_update, lr=0.01, momentum=0.9)

    num_epochs = 2

    train_model(net, dataloaders_dict, critetion, optimizer, numepochs=num_epochs)

if __name__ == '__main__':
    main()