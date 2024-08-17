import os
from random import shuffle

root = "Sonar Passivo"
main_dir = "Dados_SONAR"
new_dir = "DadosSonar"

path = os.path.join(root,main_dir)

sections = ["train","test","validate"]

def modelate(path):
    for archive in os.listdir(path):
        new_path = os.path.join(path,archive)
        try :
            os.rename(new_path,f'{root}/{main_dir}/{archive[:2]}/{archive}')    
        except FileNotFoundError :
            os.mkdir(f'{root}/{main_dir}/{archive[:2]}')
            os.rename(new_path,f'{root}/{main_dir}/{archive[:2]}/{archive}')
    
def last_bar(path) :
    for i in range(len(path),-1) :
        if path[i] == '/' : 
            print(path[:i])
            return path[:i]
    
def move(path1 , path2) :
    try:
        os.rename(path1,path2)
    except FileNotFoundError:
        os.mkdir(path2)
        move(path1,last_bar(path2))

        
for directorie in os.listdir(path):
    new_path = os.path.join(path,directorie)
    archives = os.listdir(new_path)
    
    shuffle(archives)
    for i in range(len(archives)):
        
        path2 = os.path.join(new_path,archives[i])
        move(path2,f'{root}/{new_dir}/{sections[i%3]}/{directorie}/{archives[i]}')    
