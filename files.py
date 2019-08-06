import os,os.path,re

def top_files(in_path):
    return [in_path+'/'+path_i for path_i in os.listdir(in_path)]

def multiple_dataset(in_path):
    names=bottom_files(in_path,False)
    names=[clean_str(name_i) for name_i in names]
    if(not names):
        raise Exception("No datasets at:"+in_path)
    first=names[0]
    for name_i in names[1:]:  
        if(first==name_i):
            return True
    return False

def clean_str(name_i):
    #name_i=re.sub(r'^_\D0','',name_i.strip())
    #name_i=re.sub(r0','',name_i.strip())
    digits=[ str(int(digit_i)) for digit_i in re.findall(r'\d+',name_i)]
    return "_".join(digits)

def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_sort(l):
    return sorted(l,key=natural_keys)

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