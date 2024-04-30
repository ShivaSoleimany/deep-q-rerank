from scipy.stats import kendalltau, spearmanr
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
from model.mdp import compute_reward, State
import torch.nn as nn
import torch.nn.functional as F
from util.preprocess import *
from util.helper_functions import set_manual_seed

set_manual_seed()


def calculate_MRR(qrel_file, run_file, k):

    qrel, run = {}, {}

    with open(qrel_file, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split("\t")
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    if isinstance(run_file, pd.DataFrame):
        for index, row in run_file.iterrows():
            qid = str(row['qid'])
            did = str(row['doc_id'])
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    else:
        with open(run_file, 'r') as f_run:
            for line in f_run:
                qid, _, did, *_ = line.strip().split(" ")
                if qid not in run:
                    run[qid] = []
                run[qid].append(did)
        
    mrr = 0.0
    qids = []
    rrs = []

    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                break
        qids.append(qid)
        rrs.append(rr)
        mrr += rr
    mrr /= len(run)

    return mrr

def get_feature(qid, doc_id, dataset, feature):
    df = dataset[(dataset["qid"] == qid) & (dataset["doc_id"] == doc_id)]
    return float(list(df[feature])[0])

def get_feature_for_docs(r, qid, dataset, feature):
    return [get_feature(qid, doc, dataset, feature) for doc in r]

def evaluate_ranking_list(r, qid, k, dataset):

    relevance_list = get_feature_for_docs(r, qid, dataset, "rank")
    return ndcg_at_k(relevance_list, k)

def reward_from_query(agent, qid, df):
    """
    Run agent to rank a whole (single) query
    agent: DQN agent
    qid: string query id4
    """
    filtered_df = df.loc[df["qid"] == int(qid)].reset_index()
    remaining = list(filtered_df["doc_id"])
    state = State(0, qid, remaining)
    total_reward, t= 0, 0
    while not state.terminal:
        next_action = agent.get_action(state)
        t += 1
        remaining.remove(next_action)
        state = State(t, qid, remaining)

        reward = compute_reward(t, get_feature(qid, next_action, letor, "rank"), get_feature(qid, next_action, letor, "bias"))
        
        total_reward += reward
    return total_reward


def get_agent_ranking_list(agent, qid, features, df, normalized):
    """
    Run agent to rank a whole (single) query and get list
    agent: DQN agent
    qid: string query id4
    """
    filtered_df = df.loc[df["qid"] == str(qid)].reset_index()
    remaining = list(filtered_df["doc_id"])
    random.shuffle(remaining)
    state = State(0, qid, remaining)
    ranking = []
    t = 0

    valid_expected_list = []
    while len(remaining) > 0:
        
        next_action, valid_expected = agent.get_action(state, df)
        valid_expected_list.append(valid_expected)
        
        t += 1
        remaining.remove(next_action)
        state = State(t, qid, remaining)

        #----------------
        a = filtered_df[filtered_df["doc_id"] == next_action]['relevance'].values

        model_inputs = np.array(get_model_inputs(state, next_action, features, filtered_df, normalized))
        model_inputs = torch.FloatTensor(model_inputs)
        b = agent.model.forward(model_inputs)
        loss_function = nn.MSELoss(reduction='mean')
        a = torch.tensor(a) * 10

        mse_loss = F.mse_loss(a, b).item()

        ranking.append(next_action)
    if all(valid_expected_list):
        correct_max =1 
    else:
        correct_max = 0
    return ranking, mse_loss, correct_max
    

def get_true_ranking(qid, dataset):
    """
    @qid: string query id
    @return List<doc_id strings>
    """
    df = dataset[dataset["qid"] == qid]
    df.sort_values(["rank"], inplace=True, ascending=False)
    return list(df["doc_id"])


def all_error_single(agent, k_list, qid, dataset):
    """
    Returns NDCG@k list plus tau for a single qid
    """
    agent_ranking = get_agent_ranking_list(agent, qid, dataset)
    relevance_list = get_feature_for_docs(agent_ranking, qid, dataset, "rank")
    return all_ndcg_values_plus_tau(relevance_list, k_list)

def eval_agent_final(agent, k_list, dataset):
    """
    Returns a list of average NDCG@k for each k in the k_list, plus Mean NDCG
    """
    qid_set = set(dataset["qid"])
    ndcg_list = np.append(np.zeros(len(k_list)), 0)
    for qid in qid_set:
        ndcg_list += np.array(all_ndcg_single(agent, k_list, qid, dataset))
    ndcg_list /= len(qid_set)

    # print("NDCG Values: {}".format(ndcg_list))
    return ndcg_list

def get_all_errors(agent, k_list, dataset):
    """
    Returns NDCG@k List, Kendall's Tau, and Precision @ k
    """
    qid_set = set(dataset["qid"])
    ndcg_list = np.zeros(len(k_list)+2)
    for qid in qid_set:
        ndcg_list += np.array(all_error_single(agent, k_list, qid, dataset))
    ndcg_list /= len(qid_set)
    print("NDCG Values: {}".format(ndcg_list))
    return ndcg_list
    
def write_trec_results(agent, dataset, feature_names, features,  normalized, output_file_path: str):    
    test_mse = []
    correct_max_count = 0 
    with open(output_file_path, 'a+') as file:
        # file.write(f"qid QO doc_id feature_scores ModelName rank\n")
        for qid in set(dataset["qid"]):
            
            agent_ranking, mse_loss,  correct_max= get_agent_ranking_list(agent, qid, features, dataset, normalized)
            correct_max_count += correct_max
            test_mse.append(mse_loss)
            for rank, doc_id in enumerate(agent_ranking, start=1):
                feature_scores = []
                for feature_name in feature_names:
                    feature_score = dataset[(dataset["qid"] == qid) & (dataset["doc_id"] == str(doc_id)  )][feature_name].values[0]
                    feature_scores.append(feature_score)
                feature_scores = [str(a) for a in feature_scores]
                feature_scores = " ".join(feature_scores)
                file.write(f"{qid} QO {doc_id} {feature_scores} ModelName\n")

        print(f"number of correctly learned qid:{correct_max_count}")
        
    return np.mean(test_mse)


