import os
from typing import Dict
from loguru import logger
import inspect
import datetime
import numpy as np
import random
import torch


def set_manual_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(42)

def create_directories(directories_list: Dict[str, str]) -> None:
    """
    Create directories from a list of directory paths if they do not already exist.
    Args:
        directories_list: A dictionary where keys are directory names
        and values are the corresponding directory paths.

    Returns:
        None
    """
    for directory_name, directory_path in directories_list.items():
    
        if not os.path.exists(directory_path) and not directory_path.endswith(('.txt', '.csv', '.pth', '.md')):
            print(directory_path)
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")


def format_bias_output(data):

    output_lines = []
    for i, line in enumerate(data):
        output_lines.append(line)
        if (i + 1) % 2 == 0 and i + 1 < len(data):
            output_lines.append('-' * 45)

    formatted_output = '\n'.join(output_lines)
    return formatted_output




def save_run_info(cfg, train_run_time=0, eval_run_time=0):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.isfile(cfg.metadata_file_path)

    existing_content = ""
    if file_exists:
        with open(cfg.metadata_file_path, 'r') as file:
            existing_content = file.read()

    # source_code = inspect.getsource(compute_reward)

    # source_code = source_code[:209] + str(int(cfg.reward_params.nfair_coeff)) + source_code[219:]
    # print(f"source_code:{source_code}\n----------\n")
    # source_code = source_code[:222] + str(int(cfg.reward_params.relevance_coeff)) + source_code[237:]
    # print(f"source_code:{source_code}\n----------\n")
    # compute_reward_source_code = source_code[:240] + str(int(cfg.reward_params.bias_coeff)) + source_code[250:]
    # print(f"source_code:{compute_reward_source_code}\n----------\n")

    # compute_reward_source_code = source_code[:168] + str(cfg.r) + source_code[169:184] + str(cfg.b) + source_code[185:]
    # Open the file in write mode to update its content
    with open(cfg.metadata_file_path, 'w') as file:
        file.write(f"# Run Information\n\n")
        file.write(f"**Script was run on:** {current_date}\n")
        file.write(f"**Run mode:** {cfg.run_mode}\n")
        file.write(f"**Reward Coeffs:** {cfg.reward_params}\n")
        file.write(f"**Model parameters\n:** {cfg.model_config_dict}\n")
        file.write(f"**Number of epochs:** {cfg.run_params.epochs}\n")
        file.write(f"**Train Run time:** {train_run_time} seconds\n")
        file.write(f"**Eval Run time:** {eval_run_time} seconds\n")
        # file.write(f"## Reward Function Definition\n\n")
        # file.write(f"```python\n")
        # file.write(compute_reward_source_code)
        # file.write(f"```\n")
        file.write(existing_content)
