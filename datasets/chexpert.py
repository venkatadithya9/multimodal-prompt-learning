import os
import pandas as pd
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class CheXpert(DatasetBase):
    """CheXpert dataset adapted for Dassl."""

    dataset_dir = "chexpert"

    def __init__(self, cfg):
        root = cfg.DATASET.ROOT
        print(cfg.DATASET.NAME, cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        print("DATASET_DIR is set to", self.dataset_dir)
        
        # Define classnames as a local list first
        classnames = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]

        # 1. Load the raw data into Datum lists
        train = self._read_csv(os.path.join(self.dataset_dir, 'train.csv'), classnames)
        val = self._read_csv(os.path.join(self.dataset_dir, 'valid.csv'), classnames)
        test = self._read_csv(os.path.join(self.dataset_dir, 'valid.csv'), classnames)

        # 2. Few-Shot Logic (from food101.py template)
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
            print(f"DEBUG: Attempting to create directory: {self.split_fewshot_dir}")
            mkdir_if_missing(self.split_fewshot_dir)
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 3. Call super().__init__ and pass the classnames here
        # This correctly populates the property without direct assignment
        super().__init__(train_x=train, val=val, test=test)

    def _read_csv(self, csv_path, classnames):
        print(f"DEBUG: Attempting to read CSV at: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"ERROR: File not found: {csv_path}")
            return []
                
        df = pd.read_csv(csv_path)
        print(f"DEBUG: Found {len(df)} rows in {os.path.basename(csv_path)}")
        
        items = []
        for _, row in df.iterrows():
            # Ensure path is cleaned
            rel_path = str(row['Path']).lstrip("/") 
            impath = os.path.join(self.dataset_dir, rel_path)
            
            label = int(row['label'])
            classname = classnames[label]
            
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
                
        return items