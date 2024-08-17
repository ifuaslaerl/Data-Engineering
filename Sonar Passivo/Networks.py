
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

inf = 2e18

class Sonar_CNN(nn.Module):

    def __init__(self):
        super(Sonar_CNN, self).__init__()

        self.Conv1d = nn.Conv1d(1 , NF , TK)
        self.MaxPooling1D = nn.MaxPool1d(PO)
        self.Dropout = nn.Dropout(DR)
        self.flatten = torch.flatten
        self.Dense1 = nn.Linear(121*128,NN)
        self.Dense2 = nn.Linear(NN,28)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        
        x = F.relu(self.Conv1d(x))
        x = self.MaxPooling1D(x)    
        x = self.Dropout(x)
        x = self.flatten(x,1)
        x = self.flatten(x)
        x = F.relu(self.Dense1(x))
        x = self.Dense2(x)

        return x

def fit(model,trainloader,validateloader,root):

    start = time()
    minimum=inf
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
        
        loss_in = train_loss/len(trainloader)
        
        model.eval()
        validate_loss = 0
        for data, label in validateloader:
            with torch.no_grad():
                outputs = model(data)
                loss = model.criterion(outputs, label)
                validate_loss += loss.item()

        loss_out = validate_loss/len(validateloader)

        print(f'Epoch [{epoch+1}/{NE}] - \
                Loss_in: {loss_in :.3f} - \
                Loss_out: {loss_out :.3f} - \
                in {int(time()-start) :03d} seconds')
        
        if loss_out < minimum :
            minimum = loss_out
            
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'epoch': epoch,    
            'loss_in' : loss_in,
            'loss_out': loss_out,   
            }, f'{root}/Networks/{int(1000*loss_out) :04d}.pth')
