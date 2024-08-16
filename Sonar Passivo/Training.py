#Bibliotecas

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import MatDataset
import Networks

BS = 1 # tamanho dos conjuntos trabalhados

transform = transforms.Compose([transforms.ToTensor()])

# Dataset

data = 'Sonar Passivo/Dados_SONAR'

trainset = MatDataset.MatDataset(f'{data}/train',
                        transform=transform)

testset = MatDataset.MatDataset(f'{data}/test',
                        transform=transform)

validateset = MatDataset.MatDataset(f'{data}/validate',
                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                        BS=BS,
                                        shuffle=True,
                                        num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
                                        BS=BS,
                                        shuffle=True,
                                        num_workers=2)

validateloader = torch.utils.data.DataLoader(testset,
                                        BS=BS,
                                        shuffle=True,
                                        num_workers=2)

model = Networks.Sonar_CNN()



