import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from instruct_datasets import (
    GemmaInstructDataset,
    MistralInstructDataset,
    LlamaInstructDataset,
    Llama3InstructDataset,
)

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "You are a helpful programming agent, employed at mali. Answer clearly, in context to a project"
DATASET_NAME = "mali_llama3_instruct_dataset"
DATASETS_PATHS = [
    r"../data/raw_data/csv/train.csv",
    r"../data/raw_data/csv/valid.csv",
    r"../data/raw_data/csv/test.csv"
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_dataset(dataset_path: str, model: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the model.
    :param dataset_path: The path to the dataset.
    :param model: The model to process the dataset for.
    :return: The processed dataset.
    """
    logger.info(f"Processing dataset: {dataset_path} for {model} instruct model.")
    if model == "gemma":
        dataset = GemmaInstructDataset(dataset_path)
    elif model == "mistral":
        dataset = MistralInstructDataset(dataset_path)
    elif model == "llama":
        dataset = LlamaInstructDataset(dataset_path)
    elif model == "llama3":
        dataset = Llama3InstructDataset(dataset_path)
    else:
        raise ValueError(f"Model {model} not supported!")
    dataset.drop_columns(REMOVE_COLUMNS)
    logger.info("Columns removed!")
    dataset.rename_columns(RENAME_COLUMNS)
    logger.info("Columns renamed!")
    dataset.create_instruction(INSTRUCTION)
    logger.info("Instructions created!")
    dataset.drop_bad_rows(["input", "output"])
    logger.info("Bad rows dropped!")
    dataset.create_prompt()
    logger.info("Prompt column created!")
    return dataset.get_dataset()


def create_dataset_hf(
    dataset: pd.DataFrame,
    split: str
) -> DatasetDict:
    """
    Create a Hugging Face dataset from the pandas dataframe.
    :param dataset: The pandas dataframe.
    :param split: The split of the dataset (train, test, or valid).
    :return: The Hugging Face dataset.
    """
    dataset.reset_index(drop=True, inplace=True)
    return DatasetDict({f"{DATASET_NAME}-{split}": Dataset.from_pandas(dataset)})


if __name__ == "__main__":
    processed_data_path = r"../data/processed_data"
    os.makedirs(processed_data_path, exist_ok=True)

    splits = ["train", "valid", "test"]
    datasets = {"mali_llama3_instruct_dataset": {}}

    for dataset_path, split in zip(DATASETS_PATHS, splits):
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        llama3_dataset = process_dataset(dataset_path, "llama3")
        llama3_dataset = create_dataset_hf(llama3_dataset, split)

        datasets["mali_llama3_instruct_dataset"][split] = llama3_dataset["mali_llama3_instruct_dataset-"+split]

    dataset_dict = DatasetDict(datasets["mali_llama3_instruct_dataset"])
    dataset_dict.save_to_disk(
        os.path.join(processed_data_path, "mali_llama3_instruct_dataset")
    )
    processed_data_path = r"../data/processed_data"
    os.makedirs(processed_data_path, exist_ok=True)

    splits = ["train", "valid", "test"]
    for dataset_path, split in zip(DATASETS_PATHS, splits):
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        #mistral_dataset = process_dataset(dataset_path, "mistral")
        #llama_dataset = process_dataset(dataset_path, "llama")
        llama3_dataset = process_dataset(dataset_path, "llama3")
        #gemma_dataset = process_dataset(dataset_path, "gemma")

        #mistral_dataset = create_dataset_hf(mistral_dataset, split)
        #llama_dataset = create_dataset_hf(llama_dataset, split)
        llama3_dataset = create_dataset_hf(llama3_dataset, split)
        #gemma_dataset = create_dataset_hf(gemma_dataset, split)

        # mistral_dataset.save_to_disk(
        #     os.path.join(processed_data_path, f"mali_mistral_instruct_dataset-{split}")
        # )
        # llama_dataset.save_to_disk(
        #     os.path.join(processed_data_path, f"mali_llama2_instruct_dataset-{split}")
        # )
        llama3_dataset.save_to_disk(
            os.path.join(processed_data_path, f"mali_llama3_instruct_dataset-{split}")
        )
        # gemma_dataset.save_to_disk(
        #     os.path.join(processed_data_path, f"mali_gemma_instruct_dataset-{split}")
        # )