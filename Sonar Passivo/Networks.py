
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time

NE = 55
BS = 1
NF = 128
TK = 71
PO = 4
DR = 0.5
NN = 75

class Sonar_CNN(nn.Module):

    def __init__(self):
        super(Sonar_CNN, self).__init__()

        self.Conv1d = nn.Conv1d(BS , NF , TK)
        self.MaxPooling1D = nn.MaxPool1d(PO)
        self.Dropout = nn.Dropout(DR)
        self.flatten = torch.flatten
        self.Dense = nn.Linear(NN,28)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        
        x = F.relu(self.Conv1d(x))
        x = self.MaxPooling1D(x)
        x = self.Dropout(x)
        x = self.flatten(x,1)
        x = F.relu(self.Dense(x))

        return x

def fit(model,trainloader,validateloader):

    start = time()
    for epoch in range(NE):
        
        model.train()
        train_loss = 0.0
        for data, label in trainloader:
            model.optimizer.zero_grad() 
            outputs = model(data)  
            loss = model.criterion(outputs, label)
            loss.backward() 
            model.optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss/len(trainloader)
        
        model.eval()
        validate_loss = 0
        for data, label in validateloader:
            with torch.no_grad():
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                validate_loss += (label==predicted)

        print(f'Epoch [{epoch+1}/{NE}], Loss: {avg_loss :.3f} in {int(time()-start) :03d} seconds')

'''
start = time()
epoch=1
avg_loss = 10
while avg_loss > 0.5 :

    model.train() # modo de treino do modelo
    train_loss = 0.0
    
    for images, labels in trainloader:
        optimizer.zero_grad()  # Zerar os gradientes
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calcular a perda
        loss.backward()  # Backpropagation
        optimizer.step()  # Atualizar os pesos
        train_loss += loss.item()
    
    avg_loss = train_loss/len(trainloader)
    print(f'Epoch [{epoch}], Loss: {avg_loss :.3f} in {int(time()-start) :03d} seconds')
    epoch+=1

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,    # Quantas epocas a rede foi treinada
    'loss': avg_loss,   # media de erro da rede neural
}, f'CNN/{data}_{int(1000*avg_loss) :04d}.pth')

'''