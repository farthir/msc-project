# file_handling module containing functions to handle files


def read_training(file_name):
    open_mode = 'r'
    file_object = open(file_name, open_mode)

    for line in file_object:
        print(line)
