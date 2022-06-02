from torch.utils.data import Dataset
import pandas as pd
import PIL
import os

class CustomDataset(Dataset):
    def __init__(self, classes_file, image_dir, transform=None):
        """
        Args:
            classes_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform: Set of torch transformation operations.
        """
        self.classes_file = pd.read_csv(classes_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.classes_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                               self.classes_file.iloc[idx,0])

        img = PIL.Image.open(img_name).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.classes_file.iloc[idx,2]

        return img, label
