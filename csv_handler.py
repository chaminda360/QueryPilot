import csv

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def write_csv(file_path, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
