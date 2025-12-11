#!/usr/bin/env python

import sys
import csv
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score

'''
Given a CSV file like this:
filename,predicted_label,correct_label
/home/asi094/tiles_eigth_1/TCGA-85-8287-01A-01-BS1/21505_36865.jpg,3,4
/home/asi094/tiles_eigth_1/TCGA-85-8048-01A-01-TS1/30721_52225.jpg,1,4
/home/asi094/tiles_eigth_1/TCGA-85-7699-01A-01-BS1/53761_47617.jpg,3,4

compute the accuracy and cohen's kappa
'''

def main():
    if len(sys.argv) != 2:
        print("Usage: python kappa_from_predictions.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Read CSV file and extract predictions
    predicted_labels = []
    correct_labels = []
    
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            predicted_labels.append(int(row['predicted_label']))
            correct_labels.append(int(row['correct_label']))
    
    # Compute metrics
    accuracy = accuracy_score(correct_labels, predicted_labels)
    kappa = cohen_kappa_score(correct_labels, predicted_labels)
    
    # Prepare output
    output_lines = [
        f"Results for {csv_file}:",
        f"Number of samples: {len(predicted_labels)}",
        f"Accuracy: {accuracy:.4f}",
        f"Cohen's Kappa: {kappa:.4f}"
    ]
    
    # Print results
    for line in output_lines:
        print(line)
    
    # Create output filename
    base_name = os.path.splitext(csv_file)[0]
    extension = os.path.splitext(csv_file)[1]
    output_file = f"{base_name}.predictions{extension}"
    
    # Write results to file
    with open(output_file, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()