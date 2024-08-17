#Bibliotecas

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import MatDataset
import Networks

BS = 1 # tamanho dos conjuntos trabalhados

transform = transforms.Compose([transforms.ToTensor()])

# Dataset

data = 'Sonar Passivo/DadosSonar'

trainset = MatDataset.MatDataset(f'{data}/train')

testset = MatDataset.MatDataset(f'{data}/test')

validateset = MatDataset.MatDataset(f'{data}/validate')

trainloader = torch.utils.data.DataLoader(trainset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

validateloader = torch.utils.data.DataLoader(validateset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

model = Networks.Sonar_CNN()

Networks.fit(model,trainloader,validateloader,root="Sonar Passivo")


