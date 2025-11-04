import json
import argparse
import os

def merge_datasets(input_files, output_file):
    """
    Merges multiple dataset JSON files into a single file.
    """
    merged_data = {}

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key not in merged_data:
                    merged_data[key] = []
                # Assuming value is a list of records
                if isinstance(value, list):
                    merged_data[key].extend(value)
                else:
                    # Handle cases where a key might not be a list (like 'example_data' in blank_output.json)
                    if isinstance(merged_data[key], list):
                         print(f"Warning: Mismatch in data structure for key '{key}' in {file_path}. Expected a list, but got {type(value)}. Skipping this key.")
                    elif isinstance(merged_data.get(key), dict) and isinstance(value, dict):
                        merged_data[key].update(value)
                    else:
                        merged_data[key] = value


    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    for key, value in merged_data.items():
        if isinstance(value, list):
            print(f"  - Category '{key}': {len(value)} records")

if __name__ == '__main__':

    input_files_to_merge = [
        'data/dataset_0.json',
        'data/dataset_1.json' 
    ]

    output_merged_file = 'data/merged_dataset.json'

    merge_datasets(input_files_to_merge, output_merged_file)