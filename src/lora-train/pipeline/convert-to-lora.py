import json

with open('../data/instruct/instruction_cleaned.json', 'r') as file:
    data = json.load(file)

with open('../data/instruct/instruct_alpaca.jsonl', 'w') as file:
    for obj in data['questions']:
        if 'Q' in obj and 'A' in obj:
            value = {'Q': obj['Q'], 'A': obj['A']}
            value = {k: v.replace("'", "\\'") if isinstance(v, str) else v for k, v in value.items()}
            file.write(json.dumps({'text': f'Q: {value["Q"]}\nA: {value["A"]}'}, ensure_ascii=False) + '\n')