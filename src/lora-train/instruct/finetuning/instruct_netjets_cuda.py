import json

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Defining the configuration for the base model, LoRA and training
config = {
    "hugging_face_username":"Shekswess",
    "model_config": {
        "base_model":"unsloth/llama-3-8b-Instruct-bnb-4bit", # The base model
        "finetuned_model":"llama-3-8b-Instruct-bnb-4bit-mali", # The finetuned model
        "max_seq_length": 2048, # The maximum sequence length
        "dtype":torch.float16, # The data type
        "load_in_4bit": True, # Load the model in 4-bit
    },
    "lora_config": {
      "r": 16, # The number of LoRA layers 8, 16, 32, 64
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # The target modules
      "lora_alpha":16, # The alpha value for LoRA
      "lora_dropout":0, # The dropout value for LoRA
      "bias":"none", # The bias for LoRA
      "use_gradient_checkpointing":True, # Use gradient checkpointing
      "use_rslora":False, # Use RSLora
      "use_dora":False, # Use DoRa
      "loftq_config":None # The LoFTQ configuration
    },
    "training_dataset":{
        "name":"../data/processed_data/mali_llama3_instruct_dataset", # The dataset name(huggingface/datasets)
        "split":"train", # The dataset split
        "input_field":"prompt", # The input field
    },
    "training_config": {
        "per_device_train_batch_size": 2, # The batch size
        "gradient_accumulation_steps": 4, # The gradient accumulation steps
        "warmup_steps": 5, # The warmup steps
        "max_steps":0, # The maximum steps (0 if the epochs are defined)
        "num_train_epochs": 1, # The number of training epochs(0 if the maximum steps are defined)
        "learning_rate": 2e-4, # The learning rate
        "fp16": not torch.cuda.is_bf16_supported(), # The fp16
        "bf16": torch.cuda.is_bf16_supported(), # The bf16
        "logging_steps": 1, # The logging steps
        "optim" :"adamw_8bit", # The optimizer
        "weight_decay" : 0.01,  # The weight decay
        "lr_scheduler_type": "linear", # The learning rate scheduler
        "seed" : 42, # The seed
        "output_dir" : "outputs", # The output directory
    }
}

# Loading the model and the tokinizer for the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.get("model_config").get("base_model"),
    max_seq_length = config.get("model_config").get("max_seq_length"),
    dtype = config.get("model_config").get("dtype"),
    load_in_4bit = config.get("model_config").get("load_in_4bit"),
    cuda = False,
)

# Setup for QLoRA/LoRA peft of the base model
model = FastLanguageModel.get_peft_model(
    model,
    r = config.get("lora_config").get("r"),
    target_modules = config.get("lora_config").get("target_modules"),
    lora_alpha = config.get("lora_config").get("lora_alpha"),
    lora_dropout = config.get("lora_config").get("lora_dropout"),
    bias = config.get("lora_config").get("bias"),
    use_gradient_checkpointing = config.get("lora_config").get("use_gradient_checkpointing"),
    random_state = 42,
    use_rslora = config.get("lora_config").get("use_rslora"),
    use_dora = config.get("lora_config").get("use_dora"),
    loftq_config = config.get("lora_config").get("loftq_config"),
    cuda = False,
)

# Loading the training dataset
dataset_train = load_dataset(config.get("training_dataset").get("name"), split = config.get("training_dataset").get("split"))

# Setting up the trainer for the model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    dataset_text_field = config.get("training_dataset").get("input_field"),
    max_seq_length = config.get("model_config").get("max_seq_length"),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = config.get("training_config").get("per_device_train_batch_size"),
        gradient_accumulation_steps = config.get("training_config").get("gradient_accumulation_steps"),
        warmup_steps = config.get("training_config").get("warmup_steps"),
        max_steps = config.get("training_config").get("max_steps"),
        num_train_epochs= config.get("training_config").get("num_train_epochs"),
        learning_rate = config.get("training_config").get("learning_rate"),
        fp16 = config.get("training_config").get("fp16"),
        bf16 = config.get("training_config").get("bf16"),
        logging_steps = config.get("training_config").get("logging_steps"),
        optim = config.get("training_config").get("optim"),
        weight_decay = config.get("training_config").get("weight_decay"),
        lr_scheduler_type = config.get("training_config").get("lr_scheduler_type"),
        seed = 42,
        output_dir = config.get("training_config").get("output_dir"),
    ),
)


# Memory statistics before training
gpu_statistics = torch.cuda.get_device_properties(0)
reserved_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
max_memory = round(gpu_statistics.total_memory / 1024**3, 2)
print(f"Reserved Memory: {reserved_memory}GB")
print(f"Max Memory: {max_memory}GB")

# Training the model
trainer_stats = trainer.train()


# Memory statistics after training
used_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
used_memory_lora = round(used_memory - reserved_memory, 2)
used_memory_persentage = round((used_memory / max_memory) * 100, 2)
used_memory_lora_persentage = round((used_memory_lora / max_memory) * 100, 2)
print(f"Used Memory: {used_memory}GB ({used_memory_persentage}%)")
print(f"Used Memory for training(fine-tuning) LoRA: {used_memory_lora}GB ({used_memory_lora_persentage}%)")

# Saving the trainer stats
with open("trainer_stats.json", "w") as f:
    json.dump(trainer_stats, f, indent=4)


# Locally saving the model
model.save_pretrained(config.get("model_config").get("finetuned_model"))

# Saving the model using merged_16bit(float16), merged_4bit(int4) or quantization options(q8_0, q4_k_m, q5_k_m)...
model.save_pretrained_merged(config.get("model_config").get("finetuned_model"), tokenizer, save_method = "merged_16bit",)

model.save_pretrained_merged(config.get("model_config").get("finetuned_model"), tokenizer, save_method = "merged_4bit",)

model.save_pretrained_gguf(config.get("model_config").get("finetuned_model"), tokenizer)

model.save_pretrained_gguf(config.get("model_config").get("finetuned_model"), tokenizer, quantization_method = "f16")


model.save_pretrained_gguf(config.get("model_config").get("finetuned_model"), tokenizer, quantization_method = "q4_k_m")

# Loading the fine-tuned model and the tokenizer for inference
# model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = config.get("model_config").get("finetuned_model"),
#         max_seq_length = config.get("model_config").get("max_seq_length"),
#         dtype = config.get("model_config").get("dtype"),
#         load_in_4bit = config.get("model_config").get("load_in_4bit"),
#     )

# # Using FastLanguageModel for fast inference
# FastLanguageModel.for_inference(model)

# # Tokenizing the input and generating the output
# inputs = tokenizer(
# [
#     "<|start_header_id|>system<|end_header_id|> Answer the question truthfully, you are a mali professional.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: Can you provide an overview of the lung's squamous cell carcinoma?<|eot_id|>"
# ], return_tensors = "pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
# tokenizer.batch_decode(outputs, skip_special_tokens = True)