import argparse
import os
import yaml
from pipeline import Pipeline
from typing import Literal
import models
from models import *
from modules import *
from utils import *
from modules import *
from confidence import ConfidenceCalculator
import json
from pathlib import Path
import time

pdf_files = [
    "unirs.pdf",
    "usat.pdf",
    "vhm.pdf",
    "wildsat.pdf",
    "xlrs_bench.pdf"
]

def ascii_clean(s):
    return s.encode("ascii", errors="replace").decode("ascii")

def clean_data_ascii(obj):
    if isinstance(obj, str):
        return ascii_clean(obj)
    elif isinstance(obj, list):
        return [clean_data_ascii(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data_ascii(value) for key, value in obj.items()}
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Run the extraction framefork.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--file_path', type=str, default=None,
                        help='Override the file path in the extraction config.')

    args = parser.parse_args()

    config = load_extraction_config(args.config) 

    if args.file_path:
        if 'extraction' in config and 'file_path' in config['extraction']:
            config['extraction']['file_path'] = args.file_path
            print(f"[INFO] Overriding file_path with: {args.file_path}")
        else:
            print("[WARNING] extraction.file_path not found in config; ignoring override.")

    model_config = config['model']
    if model_config['vllm_serve'] == True:
        model = LocalServer(model_config['model_name_or_path'])
    else:
        clazz = getattr(models, model_config['category'], None) 
        if clazz is None:
            print(f"Error: The model category '{model_config['category']}' is not supported.")
            return
        if model_config['api_key'] == "":
            model = clazz(model_config['model_name_or_path'])
        else:
            model = clazz(model_config['model_name_or_path'], model_config['api_key'], model_config['base_url']) 
    pipeline = Pipeline(model)
    extraction_config = config['extraction']
    if 'construct' in config:
        construct_config = config['construct']
        result, trajectory, _, _ = pipeline.get_extract_result(task=extraction_config['task'], instruction=extraction_config['instruction'], text=extraction_config['text'], output_schema=extraction_config['output_schema'], constraint=extraction_config['constraint'], use_file=extraction_config['use_file'], file_path=extraction_config['file_path'], truth=extraction_config['truth'], mode=extraction_config['mode'], update_case=extraction_config['update_case'], show_trajectory=extraction_config['show_trajectory'],
                                                               construct=construct_config, iskg=True) 
        return
    else:
        print("please provide construct config in the yaml file.")

    repeat = 3
    file_path = extraction_config['file_path']
    print(file_path)
    generations = []
    logprobs = []
    for i in range(repeat):
        result, logprob_result, trajectory, _, _ = pipeline.get_extract_result(task=extraction_config['task'], instruction=extraction_config['instruction'], text=extraction_config['text'], output_schema=extraction_config['output_schema'], constraint=extraction_config['constraint'], use_file=extraction_config['use_file'], file_path = file_path, truth=extraction_config['truth'], mode=extraction_config['mode'], update_case=extraction_config['update_case'], show_trajectory=extraction_config['show_trajectory'])
        generations.append(result)
        logprobs.append(logprob_result)
    calc = ConfidenceCalculator()
    schema_name = extraction_config['output_schema'] + "_conf"
    schema_class = getattr(schema_repository, schema_name, None)
    final = calc.process(schema_class, generations, logprobs)

    print(final)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    print(extraction_config['use_file'])
    if extraction_config['use_file']:
        print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        fm_file_path = Path(file_path)
        filename_without_ext = fm_file_path.stem
        new_filename = f"{filename_without_ext}_prediction.json"
        target_dir = Path("./model_metadata")
        new_file_path = target_dir / new_filename
        print(new_file_path)
        with open(new_file_path, "w", encoding="utf-8") as f:
            json.dump(clean_data_ascii(final), f, indent=4)
    return

if __name__ == "__main__":
    main()
