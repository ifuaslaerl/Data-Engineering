#Bibliotecas

import torch
import torchvision.transforms as transforms
import MatDataset
import Networks

NE = 75
BS = 2 # tamanho dos conjuntos trabalhados
data = 'DadosSonar'
root = ""

def main():

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    
    print(f'Using {device} device')

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

    Networks.fit(model,trainloader,validateloader,root,NE)

if __name__ == "__main__" : main()
