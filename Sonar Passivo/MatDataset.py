import torch
import scipy.io 
import os
from torch.utils.data import Dataset

class MatDataset(Dataset) :

    def __init__(self , main_path , transform=None) :
        self.main_path = main_path
        self.transform = transform
        self.classes = []
        self.data = []
        self.label = []
        
        directories = os.listdir(self.main_path)
        for i in range(len(directories)) :
            directorie = directories[i]
            sub_path = os.path.join(self.main_path,directorie)
            self.classes += [directorie]
            for archive in os.listdir(sub_path) :
                mat_data = scipy.io.loadmat(os.path.join(sub_path,archive))
                matriz = mat_data["ent_norm"]
                for line in matriz:
                    self.data += [line]
                    self.label += [(i-1)*[0] + [i] + (len(directories)-i)*[0]] # LAST CHANGE

    def __len__(self) : 
        if len(self.data) == len(self.label) :
            return len(self.data)
        return -1

    def __getitem__(self, index):
        
        data = torch.tensor(self.data[index]).float()
        label = torch.tensor(self.label[index])

        if self.transform!=None : data = self.transform(data)

        return data , label
