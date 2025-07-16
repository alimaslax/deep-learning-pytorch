import json
import csv
import random

# Load the JSON data
with open('./raw_data/instruction_cleaned.json', 'r') as file:
    data = json.load(file)['questions']  # Assuming the list is in a 'data' key

# Shuffle the data to ensure randomness
random.shuffle(data)

# Split the data into train, test, and validation sets (80% train, 10% test, 10% validation)
train_size = int(0.8 * len(data))
test_size = int(0.1 * len(data))
validation_size = len(data) - train_size - test_size

train_data = data[:train_size]
test_data = data[train_size:train_size+test_size]
validation_data = data[train_size+test_size:]

# Write the train, test, and validation sets to separate CSV files
with open('./raw_data/csv/train.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    writer.writerow('question,answer')
    for obj in train_data:
        writer.writerow([obj['Q'], obj['A']])

with open('./raw_data/csv/test.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    writer.writerow('question,answer')
    for obj in test_data:
        writer.writerow([obj['Q'], obj['A']])

with open('./raw_data/csv/valid.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    writer.writerow('question,answer')
    for obj in validation_data:
        writer.writerow([obj['Q'], obj['A']])