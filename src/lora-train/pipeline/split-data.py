import json
import random

# Load the JSONL data
with open('../data/instruct/instruct_alpaca.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# Shuffle the data to ensure randomness
random.shuffle(data)

# Split the data into train, test, and validation sets (80% train, 10% test, 10% validation)
train_size = int(0.8 * len(data))
test_size = int(0.1 * len(data))
validation_size = len(data) - train_size - test_size

train_data = data[:train_size]
test_data = data[train_size:train_size+test_size]
validation_data = data[train_size+test_size:]

# Write the train, test, and validation sets to separate JSONL files
with open('../data/instruct/train.jsonl', 'w') as file:
    for obj in train_data:
        json.dump(obj, file)
        file.write('\n')

with open('../data/instruct/test.jsonl', 'w') as file:
    for obj in test_data:
        json.dump(obj, file)
        file.write('\n')

with open('../data/instruct/valid.jsonl', 'w') as file:
    for obj in validation_data:
        json.dump(obj, file)
        file.write('\n')