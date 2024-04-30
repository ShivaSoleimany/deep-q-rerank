import torch
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

def save_and_plot_results(numbers, window, results_folder, stage):

    with open(f"{results_folder}/losses/{stage}.txt", 'w+') as f:
        f.write(str(numbers))

    plot_loss(numbers, f"{results_folder}/plots/{stage}.png", label=f"{stage}")
    plot_loss(pd.Series(numbers).rolling(window=window).mean().tolist(), f"{results_folder}/plots/{stage}_MA.png", label=f"{stage} Moving Average")
    # plot_MA_log10(numbers, window, f"{results_folder}/plots/{stage}_MALog", label=f" MA(log({stage}))")
    
    print("results_folder", results_folder)

def plot_MA_log10(numbers: List, window: int, plot_name: str, label = ""):

    plt.figure(figsize=(10, 6))

    moving_avg = np.convolve(np.log10(numbers), np.ones(window) / window, mode='valid')
    plt.plot(moving_avg)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)

def plot_loss(numbers: List, plot_name: str, label = ""):

    if torch.is_tensor(numbers):
        numbers = numbers.detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(numbers)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)


def plot_MRR(MRR_list, plot_name):

    plt.figure(figsize=(10, 6))
    
    x_values = list(range(len(MRR_list)))
    plt.plot(x_values, MRR_list, marker='o')  # 'o' is for circle markers

    plt.title('MRR value during training')
    plt.xlabel('Epoch/250')
    plt.ylabel('MRR@10 value')
    plt.savefig(plot_name)
