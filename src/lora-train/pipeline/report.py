import json
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path) as f:
    data = json.load(f)

total_questions = len(data)

print(f'Total instruction: {total_questions}')