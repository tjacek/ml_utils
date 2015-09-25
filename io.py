
def read_csv_file(path):
    lines=read_file(path)
    return map(parse_csv_line,lines)

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()
    file_object.close()
    return lines

def parse_csv_line(line):
    print(float(line.split(",")[1]))
    return map(float,line.split(","))
