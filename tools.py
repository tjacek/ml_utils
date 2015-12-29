def save_dataset(dataset,out_path):
    csv_text=str(dataset)
    text_file = open(out_path, "w")
    text_file.write(csv_text)
    text_file.close()

def to_arff(data):
    arff="@RELATION dataset\n"
    atributes=map(get_attr_header,range(data.dim))
    arff+=array_to_string(atributes)
    arff+=get_cats_header(data.cat_names)
    arff+="\n@DATA\n"
    for instance,cat in zip(list(data.X),data.y):
        arff+=to_csv_line(instance)+str(cat)+"\n"
    return arff

def get_attr_header(i):
    return "@ATTRIBUTE attr"+str(i)+" NUMERIC\n"

def get_cats_header(cats):
    cat_header="@ATTRIBUTE class {"
    for cat_i in cats:
        cat_header+=" "+str(cat_i)
    cat_header+="}\n"
    return cat_header 

def to_csv_line(array):
    array=[str(a_i) for a_i in array]
    str_vec=",".join(array)
    return str_vec
