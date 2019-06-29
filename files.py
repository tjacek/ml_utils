import os,os.path,re

def top_files(in_path):
    return [in_path+'/'+path_i for path_i in os.listdir(in_path)]

def bottom_files(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            paths=[ root+'/'+filename_i 
                for filename_i in filenames]
            all_paths+=paths
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()
    file_object.close()
    return lines