import numpy as np
import pandas as pd
import random
from collections import deque
from loguru import logger
from util.preprocess import *
from util.helper_functions import set_manual_seed

set_manual_seed()

def calculate_MRR_reward(rank_list):

    relevant_index = next((i for i, rank in enumerate(rank_list) if rank == 1), None)
    return 1 / (relevant_index + 1) if relevant_index is not None else 0.0

def compute_reward(t, features_coeffs, sorted_list=None, reward_mode="_"):

    if reward_mode =="relevance":
        return (features_coeffs[0][0] * features_coeffs[0][1])
    elif reward_mode == "relevance_t":
        return (features_coeffs[0][0] * features_coeffs[0][1])/ t
    elif reward_mode == "MRR":
        relevance_scores = [x[1] for x in sorted_list or []]
        return calculate_MRR_reward(relevance_scores)
    else:
        return sum(feature * coeff for feature, coeff in features_coeffs)
    
class State:

    def __init__(self, t, query, remaining, sorted_list=None):

        self.t = t
        self.qid = query
        self.remaining = remaining
        self.sorted_list = sorted_list or []

    def pop(self):
        return self.remaining.pop()

    def initial(self):
        return self.t == 0

    def terminal(self):
        return len(self.remaining) == 0

    def __str__(self):
        return f"t:{self.t}, qid:{self.qid}, sorted_list:{self.sorted_list}"

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def push_batch(self, qid_list_path, df, reward_params, n):

        reward_mode = reward_params["reward_mode"]
        relevance_coef = reward_params["relevance_coeff"]
        bias_coef = reward_params["bias_coeff"]
        nfair_coef = reward_params["nfair_coeff"]

        for i in range(n):

            random_qid = random.choice(list(df["qid"]))
            filtered_df = df.loc[df["qid"] == str(random_qid)].reset_index()

            row_order = [x for x in range(len(filtered_df))]
            X = [x[1]["doc_id"] for x in filtered_df.iterrows()]
            sorted_list = []

            random.shuffle(row_order)
            for t,r in enumerate(row_order): 
                cur_row = filtered_df.iloc[r]
                old_state = State(t, cur_row["qid"], X[:], sorted_list[:])
                action = cur_row["doc_id"]
                X.remove(action)
                sorted_list.append([action, cur_row["relevance"]])

                new_state = State(t+1, cur_row["qid"], X[:], sorted_list[:])

                features_coeffs = [
                                (cur_row.get("relevance", 0), relevance_coef),
                                (cur_row.get('bias', 0), bias_coef),
                                (cur_row.get("nfair", 0), nfair_coef)
                            ]
                reward = compute_reward(t+1, features_coeffs, sorted_list[:], reward_mode)

                self.push(old_state, action, reward, new_state, t+1 == len(row_order))
                filtered_df.drop(filtered_df.index[[r]])


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)