from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/multi-20x20-masked-less-data"
    subsample_size: Optional[int] = None

height = 20
width = 20

def replace_character(s: str):
    s = s.replace('_', ':')
    s = s.replace('*', ';')
    s = s.replace('.', '<')
    return s

def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []
    
    with open(f"multi_data/{height}×{width}-masked-less-data/{set_name}.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            assert len(q) == height*width and len(a) == height*width
            inputs.append(np.frombuffer(replace_character(q).encode(), dtype=np.uint8).reshape(height, width) - ord('0'))
            labels.append(np.frombuffer(replace_character(a).encode(), dtype=np.uint8).reshape(height, width) - ord('0'))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for inp, out in zip(tqdm(inputs), labels):
        for aug_idx in range(8 if (set_name == "train") else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        # Push group
        results["group_indices"].append(puzzle_id)
        
    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 12))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=height*width,
        vocab_size=13 + 1,  # PAD + "0" ... "9" + "_" + "*" + "."
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
