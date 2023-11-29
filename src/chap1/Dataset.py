import torch.utils.data as data
from PIL import Image
class HymenopteraDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # read
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # proccessing
        img_transformed = self.transform(img, self.phase)

        # get label from filename
        label = img_path.split('/')[-2]
        # transform label to num
        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label