#Bibliotecas
import torch
import torchvision.transforms as transforms
#include Networks
#include MatDataset

BS = 1 # tamanho dos conjuntos trabalhados
data = '/content/DadosSonar'

def main():

  trainset = MatDataset(os.path.join(data,'train'))

  testset = MatDataset(os.path.join(data,'test'))

  validateset = MatDataset(os.path.join(data,'validate'))

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

  model = Sonar_CNN()

  fit(model,trainloader,validateloader,root="Sonar Passivo")
