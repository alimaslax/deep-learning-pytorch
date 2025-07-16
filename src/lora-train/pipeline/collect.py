import json
import re
import random
import ollama
import os

def split_file_into_parts(filename):
    with open(filename, 'r') as file:
        text = file.read()
        lines = text.split('\n')
        part_size = len(lines) // 5
        parts = []
        for i in range(5):
            start = i * part_size
            end = (i + 1) * part_size if i < 4 else len(lines)
            parts.append('\n'.join(lines[start:end]))
        return parts

def process_files_in_dir(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            for _ in range(5):  # Loop 5 times
                parts = split_file_into_parts(os.path.join(directory, filename))
                while parts:
                    part = random.choice(parts)
                    parts.remove(part)
                    response = ollama.chat(model='llama3:instruct', messages=[
                    {
                        'role': 'user',
                        'content': 'OUTPUT a valid json of Question and Answer about the context. [context]\n' + '\n==========\n' + part + '\n==========\n' + 'convert the above to question and answer',
                    },
                    ])

                    # Open the text file and read the contents
                    data = response['message']['content']

                    # Use regular expressions to extract the question and answer
                    questions = re.findall(r'"(.*?)":\s*"(.*?)"', data)

                    # Create a dictionary to store the questions list
                    new_output = {"questions": []}

                    # Loop through the questions and answers
                    for i in range(0, len(questions), 2):
                        if i+1 < len(questions):  # Check if the next index is within bounds
                            question = questions[i][1]
                            answer = questions[i+1][1]
                            new_output["questions"].append({"question": question, "answer": answer})

                    # Open the existing output.json file and load the data
                    try:
                        with open('output.json', 'r') as file:
                            existing_output = json.load(file)
                    except FileNotFoundError:
                        existing_output = {"questions": []}

                    # Append the new questions to the existing output
                    existing_output["questions"].extend(new_output["questions"])

                    # Open the output.json file and write the updated data
                    with open('output.json', 'w') as file:
                        json.dump(existing_output, file, indent=4)

# Call the function with the directory path
process_files_in_dir('../data/instruct/raw/')