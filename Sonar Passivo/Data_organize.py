import os
from random import shuffle

root = ""
main_dir = "/content"
new_dir = "DadosSonar"

path = os.path.join(root,main_dir)

sections = ["train","test","validate"]

def modelate(path):
    for archive in os.listdir(path):
        new_path = os.path.join(path,archive)
        if new_path.endswith('.mat') :
          try :
            os.rename(new_path,f'{main_dir}/{archive[:2]}/{archive}')    
          except FileNotFoundError :
            os.mkdir(f'{main_dir}/{archive[:2]}')
            os.rename(new_path,f'{main_dir}/{archive[:2]}/{archive}')
    
def move(path1 , path2):
        
    for i in range(len(path2)):
        if path2[i] == '/' and not os.path.exists(path2[:i]) :
            os.mkdir(path2[:i])
    os.rename(path1,path2)
        
def sort_data(main_dir,new_dir,sections):
    
    for directorie in os.listdir(main_dir):
        new_path = os.path.join(main_dir,directorie)
        
        if len(directorie) != 2 : continue
        if 'A' > directorie[0] or directorie[0] > 'Z' : continue 
        if '0' > directorie[1] or directorie[1] > '9' : continue   
        
        archives = os.listdir(new_path)
        shuffle(archives)
        for i in range(len(archives)):

            path2 = os.path.join(new_path,archives[i])
            move(path2,f'{new_dir}/{sections[i%3]}/{directorie}/{archives[i]}')    

if __name__ == "__main__" :
    #modelate(main_dir)
    sort_data(main_dir,new_dir,sections)
