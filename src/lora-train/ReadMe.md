# Curated Data
Steps:
1. Run collect.py
2. Clean the data using clean.py
3. report the data
4. convert-to-lora.py
5. split-data.py



## collect.py
This script is a data processing pipeline that takes a directory of text files, processes each file, and generates a JSON file containing questions and answers based on the content of the text files. 

Here's a step-by-step explanation:

Splitting files into parts

The split_file_into_parts function takes a text file and splits it into 5 equal parts. This is done to process smaller chunks of text, which is useful for QLORA's instruction-following capabilities.

Processing files in a directory

The process_files_in_dir function iterates over all text files in a specified directory. For each file, it:
Splits the file into 5 parts (as described above).
Loops 5 times, processing each part separately.
For each part, it:

Sends a prompt to the ollama chat model (LLaMA3) to generate a JSON output containing questions and answers based on the text part.

Extracts the questions and answers from the response using regular expressions.

Appends the new questions to an existing JSON file (output.json) or creates a new one if it doesn't exist.

## clean.py
This script is used to preprocess the instruction data by removing duplicate questions. This is important because QLORA aims to learn a mapping from instructions to outputs, and duplicate questions can lead to biased or overfitted models.
By removing duplicates, the script ensures that each question is unique, which helps to:
Reduce data redundancy
Improve model generalization
Enhance the quality of the instruction data
The cleaned data is then used as input for QLORA optimization, where the goal is to learn an optimal mapping from instructions to outputs.

```bash
python clean ../data/instruct/instruction.json
```
## convert-alpaca.py
This script assumes that your JSONL file is in the same format as the one you provided earlier, with each line containing a JSON object with a "text" key that contains another JSON object with "instruction", "input", and "output" keys.
Here's what the script does:
Loads the JSONL data into a list of Python dictionaries.
Shuffles the data to ensure randomness.
Splits the data into train, test, and validation sets using the specified proportions (80% train, 10% test, 10% validation).
Writes each set to a separate JSONL file.
Note that this script assumes that the data is already preprocessed and cleaned. If your data requires additional preprocessing, you may need to modify the script accordingly.

```bash
python convert-alpaca ../data/instruct/instruction_cleaned.json
```

```bash
python collection-scripts/clean.py
{
    "questions": [
        {
            "Q": "",
            "A": ""
        },
    ]
}
```


## visualize.py
The script produces a graph with the iteration number on the x-axis and the training loss on the y-axis. The graph shows the training loss at each iteration, which is a measure of how well the model is performing on the training data.
In the context of QLoRa, this graph is likely showing the training progress of a QLoRa model. QLoRa is a type of neural network architecture used for natural language processing tasks, and this script is likely used to train and evaluate a QLoRa model.
The two lines in the graph represent the training loss at each iteration. The line is likely a moving average of the training loss, which helps to smooth out the noise in the data and show the overall trend.
Here are some observations about the graph:
The training loss starts high and decreases as the iteration number increases, which indicates that the model is learning and improving over time.
The training loss fluctuates up and down, which is normal during training. The fluctuations may be due to the random initialization of the model's weights, the random sampling of training data, or the optimization algorithm used to update the model's weights.
The training loss seems to level off around iteration 100, which may indicate that the model has converged and is no longer improving.
Overall, this graph provides a useful visualization of the training progress of a QLoRa model, and can help developers understand how well the model is performing and whether it needs further training or tuning.

### resources (Batch size of 4 on 32GB M2 Pro)

Typically, for InstructQA, a validation loss of around 1.5 or lower is considered decent, and a loss of around 1.0 or lower is considered good.

As for the training loss, it's great to see it decreasing over time, which indicates that your model is learning and improving. A training loss of around 1.0 or lower is a good sign.
Keep in mind that the exact loss values can vary depending on the specific model architecture, hyperparameters, and training settings. But overall, your losses are heading in the right direction!

Keep training and fine-tuning your model, and you might see even better results!

```bash
python visualize.py
```
## MLX QLora
Move the converted training data to train/lora/data
```bash
resume training
python lora.py --model llama3-8B-Instruct-4Bit --train --iters 600 --resume-adapter-file adapters.npz
```

inference
```bash
python lora.py --model ../../../models/llama3-8B-Instruct-4Bit \
--adapter-file ./trainingRuns/3000iter/instruct-alpaca.npz \
--max-tokens 50 \
--prompt "ls command"
```
[Budget Fine-Tuning](https://mlops.community/budget-instruction-fine-tuning-of-llama-3-8b-instructon-medical-data-with-hugging-face-google-colab-and-unsloth/)
## Training runs
```
{"text": "Q: "}

```
## Data collection

## python Troubleshooting

```bash
requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/models/meta-llama/Meta-Llama-3-8B-Instruct/revision/main (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)')))"), '(Request ID: eeda1c2c-7d08-49e2-8610-3e86381319cc)')
```

@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}