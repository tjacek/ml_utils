
def read_csv_file(path):
    lines=read_file(path)
    return map(parse_csv_line,lines)

def read_labeled(path):
    lines=read_file(path)
    return map(parse_labeled_line,lines)

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()
    file_object.close()
    return lines

def parse_csv_line(line):
    return map(float,line.split(","))

def parse_labeled_line(line): 
    raw=line.split(',#')
    data=parse_csv_line(raw[0])
    category=int(raw[1])
    return data,category
