import json

def clean_json_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Remove objects with the same "question" value
    cleaned_data = []
    questions = set()
    for obj in data["questions"]:  # Access the "questions" key
        question = obj["question"]
        if question not in questions:
            # Rename the keys
            obj["Q"] = obj.pop("question")
            if "answer" in obj:
                obj["A"] = obj.pop("answer")
            cleaned_data.append(obj)
            questions.add(question)

    with open(output_filename, 'w') as file:
        json.dump({"questions": cleaned_data}, file, indent=4)

# Call the function with the input and output filenames
clean_json_file('./data/instruct/instruction.json', './data/instruct/instruction_cleaned.json')