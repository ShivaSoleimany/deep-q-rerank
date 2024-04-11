import argparse
import os
import time
from loguru import logger
from hydra import compose, initialize
import torch

from util import *
from model.dqn import DQN, DQNAgent

set_manual_seed()

def eval_model(eval_cfg, test_set=None):

    logger.info(f"Testing in {eval_cfg.run_mode} mode")

    trec_mode = "neutral" if len(test_set) % 215 != 0 else "social"
    trec_output_file_path = eval_cfg.eval_trec_output_file_path if trec_mode == "neutral" else eval_cfg.eval_social_trec_output_file_path

    if test_set is None or test_set.empty:
        test_set = load_dataset(eval_cfg)

    input_dim = eval_cfg.model_config_dict.input_dim
    output_dim = eval_cfg.model_config_dict.output_dim
    model_size = eval_cfg.model_config_dict.model_size
    model = DQN(input_dim, output_dim, model_size)

    MRR10_input = calculate_MRR(eval_cfg.qrel_file, test_set, 10)

    models_dir = os.path.dirname(eval_cfg.pretrained_model_path)
    model_files = sorted(
        [f for f in os.listdir(models_dir) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else float('inf')
    )

    MRR_list = []
    
    for model_file in model_files[::-1]:
        print(f"model_file:{model_file}")
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path))
        agent = DQNAgent(dataset=None, buffer=None, config_dict = eval_cfg.model_config_dict, pre_trained_model=model)

        MRR10_output = evaluate_and_log_performance(agent, test_set, eval_cfg.qrel_file, model_file, trec_mode, MRR10_input, trec_output_file_path, eval_cfg.eval_output_file_path)
        MRR_list.append(MRR10_output)
        print("---------------------------------")

    plot_MRR(MRR_list[::-1], f"{eval_cfg.plot_folder}/MRR_{trec_mode}.png")

def evaluate_and_log_performance(agent, test_set, qrel_file, model_file, trec_mode, MRR10_input, trec_output_file_path, eval_output_file_path):

    trec_output_file_path = "".join(trec_output_file_path.split(".")[0])
    current_trec_output_file_path = f"{trec_output_file_path}_{trec_mode}_{model_file}.txt"

    write_trec_results(agent, test_set, ["relevance"], current_trec_output_file_path )

    MRR10_output = calculate_MRR(qrel_file, current_trec_output_file_path, 10)

    bias_values = calculate_bias(current_trec_output_file_path)
    formatted_bias_values = format_bias_output(bias_values) 

    os.remove(current_trec_output_file_path)

    with open(eval_output_file_path, "a+") as f:
        f.write(f"input MRR@10 Value: {MRR10_input}\n")
        f.write(f"{model_file} MRR@10 Value: {MRR10_output}\n")
        f.write(f"output bias Value:\n{formatted_bias_values}\n")
        f.write(f"=========================================================\n")

    return MRR10_output

def main():

    parser = argparse.ArgumentParser(description="Running eval_script")
    parser.add_argument("--conf", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.conf:
        config_file = args.conf
        logger.info(f"Config file name: {config_file}")
    else:
        logger.info(
            "Please provide the name of the config file using the --conf argument. \nExample: --conf rank.yaml"
        )

    initialize(config_path="config")
    cfg = compose(config_name=f"{config_file}")

    start_time = time.time()
    eval_model(cfg.eval_config)
    end_time = time.time()

    eval_run_time = end_time - start_time
    save_run_info(cfg.train_config, 0, eval_run_time)

    logger.info("Finished Evaluating Model Successfully in {eval_run_time} seconds.")

if __name__ == "__main__":
    main()
