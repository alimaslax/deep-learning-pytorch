from mlx_lm import load, generate

model, tokenizer = load("../models/llama3-8B-Instruct-4Bit")
response = generate(model, tokenizer, prompt="System: You are a helpful AI assistant that gives Mac OSX Bash commands, no other text. User: give the list dir command. Assistant:", verbose=True)
