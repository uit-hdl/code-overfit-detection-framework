import csv
import os
import glob
import random
from io import TextIOWrapper
from pathlib import Path

def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        print(f"mkdir: '{dest_dir}'")

def add_dir(directory):
    all_data = []
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            all_data.append({"q": filename, "k": filename, 'filename': filename})
    return all_data

def build_file_list(data_dir, file_list_path):
    if not os.path.exists(file_list_path):
        print ("File list not found. Creating file list in {}".format(file_list_path))
        number_of_slides = len(glob.glob(f"{data_dir}{os.sep}*"))
        all_data = []
        group_by_patient = {}
        for i, directory in enumerate(glob.glob(f"{data_dir}{os.sep}*")):
            all_data += add_dir(directory)
        for d in all_data:
            patient = d['filename'].split(os.sep)[-2]
            if patient not in group_by_patient:
                group_by_patient[patient] = []
            group_by_patient[patient].append(d)

        train_data, val_data, test_data = [], [], []
        patients = list(group_by_patient.keys())
        # impose (a more) random ordering
        random.shuffle(patients)
        splits = lambda x: [int(x * 0.7), int(x * 0.1), int(x * 0.2)]
        splits = splits(len(patients))
        for i, patient in enumerate(patients):
            d = group_by_patient[patient]
            if i < splits[0]:
                train_data += d
            elif i < splits[0] + splits[1]:
                val_data += d
            else:
                test_data += d

        if not train_data:
            raise RuntimeError(f"Found no data in {data_dir}")

        ensure_dir_exists(file_list_path)
        #with open(file_list_path, 'w', newline='') as csvfile:
        with open(file_list_path, 'wb') as csvfile, TextIOWrapper(csvfile, encoding='utf-8', newline='') as wrapper:
            #csvwriter = csv.writer(csvfile)
            csvwriter = csv.writer(wrapper)
            csvwriter.writerow(["q", "k", "filename", "mode"])
            for d in train_data:
                csvwriter.writerow([d['q'], d['k'], d['filename'], "train"])
            for d in val_data:
                csvwriter.writerow([d['q'], d['k'], d['filename'], "validation"])
            for d in test_data:
                csvwriter.writerow([d['q'], d['k'], d['filename'], "test"])

    with open(file_list_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        train_data, val_data, test_data = [], [], []
        for row in csvreader:
            if row[3] == "train":
                train_data.append({"q": row[0], "k": row[1], "image": row[0], 'filename': row[2]})
            elif row[3] == "validation":
                val_data.append({"q": row[0], "k": row[1], "image": row[0], 'filename': row[2]})
            else:
                test_data.append({"q": row[0], "k": row[1], "image": row[0], 'filename': row[2]})
        print("Loaded file list from {}".format(file_list_path))
    return train_data, val_data, test_data

