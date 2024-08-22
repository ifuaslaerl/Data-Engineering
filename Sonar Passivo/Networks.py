
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time

NF = 128
TK = 71
PO = 4
DR = 0.5
NN = 75

inf = 2e18

class Sonar_CNN(nn.Module):

    def __init__(self):
        super(Sonar_CNN, self).__init__()

        self.conv1d = nn.Conv1d(1 , NF , TK)
        self.maxpooling1d = nn.MaxPool1d(PO)
        self.dropout = nn.Dropout(DR)
        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
                    nn.Linear(121*128,NN),
                    nn.ReLU(),
                    nn.Linear(NN,28)
                )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        
        x = F.relu(self.conv1d(x))
        x = self.maxpooling1d(x)    
        x = self.dropout(x)
        x = self.flatten(x)

        logits = self.dense(x)

        return logits

def train_loop(model,trainloader) :

    model.train()
    for batch, (X, y) in enumerate(trainloader) :

        pred = model(X)
        loss = model.criterion(pred, y)
        
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

    return loss

def validate_loop(model,validateloader) : 

    model.eval()
    validate_loss, correct = 0 , 0

    with torch.no_grad() :
        for X , y in validateloader:
                pred = model(X)
                validate_loss += model.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return validate_loss , correct/len(validateloader.dataset)

def fit(model,trainloader,validateloader,root,NE):

    start = time()
    minimum=inf
    for epoch in range(NE):
    
        loss_in = train_loop(model,trainloader)
        loss_out , accuracy = validate_loop(model,validateloader)

        print(f'Epoch [{epoch+1}/{NE}] - Loss_in: {loss_in :.3f} - Loss_out: {loss_out :.3f} - Accuracy: {(100*accuracy):>0.1f}% - in {int(time()-start) :03d} seconds')
        
        if loss_out < minimum :
            minimum = loss_out
            
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'epoch': epoch,    
            'loss_in' : loss_in,
            'loss_out': loss_out,   
            'accuracy' : accuracy,
            }, f'{root}/Networks/{int(1000*loss_out) :04d}.pth')
