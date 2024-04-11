import os
import time
import subprocess
import argparse
from loguru import logger
from scripts import *
from hydra import compose, initialize
from loguru import logger

from util.preprocess import *
from util.helper_functions import set_manual_seed, create_directories, save_run_info

set_manual_seed()

def main(config_files):
    
    train_set, val_set = pd.DataFrame(), pd.DataFrame()
    neutral_test_set, social_test_set, dev_test_set  = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    initialize(config_path="config")
    for config_file in config_files:
        
        cfg = compose(config_name=f"{config_file}")

        create_directories(
            cfg.directories
        )
        logger.info(f"Loading train_set for {config_file}")
        if train_set.empty:
            start_time = time.time()
            train_set = load_dataset(cfg.train_config, "TRAIN")
        logger.info(f"Loaded train_set for {config_file} in {time.time() - start_time} seconds")

        logger.info(f"Loading val_set for {config_file}")

        print(train_set.info())
        print(train_set.head(2))

        val_set = train_set #REMOVEEEE LATER
        if val_set.empty:
            start_time = time.time()
            val_set = load_dataset(cfg.valid_config, "VALID")
        logger.info(f"Loaded val_set for {config_file} in {time.time() - start_time} seconds")

        train_start_time = time.time()
        train_model(cfg, train_set, train_set)
        train_end_time = time.time()
        train_run_time = train_end_time - train_start_time
        # train_run_time = 0
        #--------------
        logger.info(f"Loading neutral neutral_test_set for {config_file}")
        if neutral_test_set.empty:
            start_time = time.time()
            neutral_test_set = load_dataset(cfg.eval_config, "EVAL1")
        logger.info(f"Loaded neutral_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, neutral_test_set)
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)
        #---------------
        logger.info(f"Loading social neutral test_set for {config_file}")
        if social_test_set.empty:
            start_time = time.time()
            social_test_set = load_dataset(cfg.eval_config, "EVAL2")
        logger.info(f"Loaded social_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, social_test_set)
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)
        #---------------
        logger.info(f"Loading social dev_test_set  for {config_file}")
        if dev_test_set.empty:
            start_time = time.time()
            dev_test_set = load_dataset(cfg.eval_config, "EVAL3")
        logger.info(f"Loaded dev_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, dev_test_set)
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config_files', metavar='N', type=str, nargs='+',
                        help='config files to be processed')
    args = parser.parse_args()

    main(args.config_files)