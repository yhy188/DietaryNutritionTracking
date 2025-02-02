import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        # F:\pytorch-unet-master\pytorch-unet-master\data\SegmentationClass
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)  # Return the count of filenames.

    def __getitem__(self, index):  # Data is centered.
        segment_name=self.name[index]  # xx.png ；Get the name (label) of the data.
        # Concatenate the address of the original image with the address of the label.
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)
        image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))
        segment_image=keep_image_size_open(segment_path)  # Resize images proportionally to the same size.
        image=keep_image_size_open(image_path)  # Maintain the same size.
        return transform(image),transform(segment_image)


if __name__ == '__main__':
    data=MyDataset('G:\pytorch-unet-master\pytorch-unet-master\data') # Your Data Address
    print(data[0][0].shape)
    print(data[0][1].shape)
