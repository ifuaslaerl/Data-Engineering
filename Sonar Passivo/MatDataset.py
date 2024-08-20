import torch
import scipy.io 
import os
from torch.utils.data import Dataset

class MatDataset(Dataset) :

    def __init__(self , main_path , transform=None) :
        self.main_path = main_path
        self.transform = transform
        self.data = []
        self.label = []

        self.classes = sorted(os.listdir(main_path))
        
        for idx , class_name in enumerate(self.classes) :
            class_path = os.path.join(main_path,class_name) 

            for archive in os.listdir(class_path) :
                mat_data = scipy.io.loadmat(os.path.join(class_path,archive))
                matriz = mat_data["ent_norm"].astype('float32')
                for line in matriz:
                    self.data.append(line)
                    self.label.append(idx)

    def __len__(self) : 
        return len(self.data)

    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]

        label = torch.tensor(label,dtype=torch.long)

        if self.transform : data = self.transform(data)

        return data , label
