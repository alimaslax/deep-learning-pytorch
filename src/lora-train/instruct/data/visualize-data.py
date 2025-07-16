from datasets import load_from_disk

# Load the dataset from the .arrow file
dataset = load_from_disk("../data/processed_data/mali_llama3_instruct_dataset")

# Print the first few rows of the 'train' split
print(dataset['valid'][:1])