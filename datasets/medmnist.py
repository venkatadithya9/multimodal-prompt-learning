import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

# Assuming MedMNIST is installed
import medmnist
from medmnist import INFO

from PIL import Image

@DATASET_REGISTRY.register()
class MedMNIST(DatasetBase):
    """MedMNIST dataset adapted for Dassl."""

    dataset_dir = "medmnist"

    def __init__(self, cfg):
        # Force correct root for now to bypass config issues
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        self.data_flag = 'pathmnist' 
        info = INFO[self.data_flag]
        self.DataClass = getattr(medmnist, info['python_class'])
        
        # Directory to save extracted images
        self.image_dir = os.path.join(self.dataset_dir, self.data_flag)
        mkdir_if_missing(self.image_dir)

        # Read data
        train = self._read_data(split='train')
        val = self._read_data(split='val')
        test = self._read_data(split='test')

        super().__init__(train_x=train, val=val, test=test)
        
    def _read_data(self, split):
        # This loads the .npz file into memory
        dataset = self.DataClass(split=split, download=True, root=self.dataset_dir)
        info = INFO[self.data_flag]
        class_names = [info['label'][str(i)] for i in range(len(info['label']))]
        
        # We create a folder to hold the images extracted from the .npz
        split_dir = os.path.join(self.image_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        items = []
        print(f"Extracting {split} images from .npz to {split_dir}...")
        
        for i in range(len(dataset)):
            img_path = os.path.join(split_dir, f"{i}.png")
            
            # Only save if not already extracted
            if not os.path.exists(img_path):
                # dataset.imgs[i] is the numpy array from the .npz file
                img = Image.fromarray(dataset.imgs[i])
                img.save(img_path)
            
            # MedMNIST labels are usually [N, 1], so we flatten to int
            label = int(dataset.labels[i])
            classname = class_names[label]
            
            item = Datum(
                impath=img_path, # This is the string Dassl needs
                label=label,
                classname=classname
            )
            items.append(item)
        return items