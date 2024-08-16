import scipy.io 
import os
from torch.utils.data import Dataset

class MatDataset(Dataset) :

    def __init__(self , main_path , transform=None) :
        self.main_path = main_path
        self.transform = transform
        self.classes = []
        self.paths = []
        self.label = []
        
        for directorie in os.listdir(self.main_path) :
            sub_path = os.path.join(self.main_path,directorie)
            self.classes += [directorie]
            for archive in os.listdir(sub_path) :
                self.paths += [os.path.join(sub_path,archive)]
                self.label += [directorie]

    def __len__(self) : 
        if len(self.paths) == len(self.label) :
            return len(self.paths)
        return -1

    def __getitem__(self, index):
        path = self.paths[index]
        
        mat_data = scipy.io.loadmat(path)

        data = mat_data['ent_norm']
        label = self.paths[index]

        if self.transform!=None : data = self.transform(data)

        return data , label
