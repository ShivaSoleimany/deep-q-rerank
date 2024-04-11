import os 
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder

from .helper_functions import set_manual_seed
set_manual_seed()


device = 'cuda:0'

def load_queries(queries_path):

    queries = {}
    with open(queries_path, 'r', encoding='utf8') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query.strip()
    return queries


def load_corpus(corpus_file_path):

    corpus = {}
    with open(corpus_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage.strip()

    return corpus

def get_vector(row, queries, corpus, cross_encoder, model):

    qid, doc_id = row[0], row[1] #str, str

    text = f'{queries[qid]}[SEP]{corpus[doc_id]}'
    tokens = cross_encoder.tokenizer(text, return_tensors="pt").to(device)
    vector = model(**tokens).pooler_output.detach().cpu().numpy().tolist()[0]
    return vector


def encode_date(queries, corpus, df, base_language_model_path):

    cross_encoder = CrossEncoder(base_language_model_path, device=device)
    model = cross_encoder.model.bert.to(device)

    enc = df.apply(get_vector, args=(queries, corpus, cross_encoder, model), axis=1, result_type='expand')
    
    df = pd.concat([df, enc], axis=1)

    return df

def load_dataset(cfg, stage= None) -> pd.DataFrame:

    if not stage:
        stage = cfg.stage
    if stage == 'TRAIN':
        input_file_path, df_path, queries_path = cfg.train_set_path, cfg.train_df_path, cfg.train_queries_path
    elif stage == 'VALID':
        input_file_path, df_path, queries_path = cfg.valid_set_path, cfg.valid_df_path, cfg.valid_queries_path
    elif stage == "EVAL1":
        input_file_path, df_path, queries_path = cfg.neutral_test_set_path, cfg.neutral_test_df_path, cfg.test_queries_path
    elif stage == "EVAL2":
        input_file_path, df_path, queries_path = cfg.social_test_set_path, cfg.social_test_df_path, cfg.test_queries_path
    else:
        input_file_path, df_path, queries_path = cfg.dev_test_set_path, cfg.dev_test_df_path, cfg.test_queries_path



    if not os.path.exists(df_path):

        df = pd.read_csv(input_file_path, names = list(cfg.run_params.columns))

        df["qid"] = df["qid"].astype(str)
        df["doc_id"] = df["doc_id"].astype(str)

        queries = load_queries(queries_path)
        corpus = load_corpus(cfg.corpus_file_path)

        common_qid = set(queries.keys()).intersection(set(df["qid"].values))
        # print(f"common_qid:{common_qid}")

        # base_language_model_path = f"/home/sajadeb/LLaMA_Debiasing/CrossEncoder/output/cross-encoder_bert-base-uncased/"
        df = encode_date(queries, corpus, df, cfg.cross_encoder_path)

        df.columns = ['qid', 'doc_id', 'relevance'] + [str(i) for i in range(1, 769)]

        df.to_csv(df_path)

    else:

        df = pd.read_csv(df_path)  
        df["qid"] = df["qid"].astype(str)
        df["doc_id"] = df["doc_id"].astype(str)

        if 'bias' in df.columns and df["bias"].dtype == "object":
            df['bias'] = df['bias'].apply(eval)
            df['bias'] = df['bias'].apply(lambda x:x[0])


        columns = list(cfg.run_params.columns) + [str(i) for i in range(1, cfg.run_params.vector_size+1)]
        df = df[columns]
        df = df.sort_values(["qid", "relevance"], ascending=False)

        logger.info(f"{stage}\n{df.head(5)}")

    return df


def get_features(qid, doc_id, dataset) -> List[float]:

    qid, doc_id = str(qid), str(doc_id)

    df = dataset[(dataset["doc_id"].str.contains(doc_id)) & (dataset["qid"] == qid)]
    assert len(df) != 0, "Fix the dataset"

    if 120 < len(df.columns) < 200:
        vector_size = 128
    elif 200 < len(df.columns) < 300:
        vector_size = 256
    else:
        vector_size = 768

    relevant_columns = [f"{i}" for i in range(1, vector_size+1)]

    # relevant_columns = relevant_columns + ["relevance", "bias", "nfair"]
    relevant_columns = relevant_columns + ["relevance"]

    return df[relevant_columns].values.tolist()[0]

def get_query_features(qid, doc_list, dataset) -> np.ndarray:
    """
    Get query features for the given query ID, list of docs, and dataset.
    """
    doc_set = set(doc_list)
    qid = str(qid)
    if len(doc_list) > 0:
        df = dataset[dataset["qid"] == qid]
        df = df[df["doc_id"].isin(doc_set)]
    else:
        df = dataset[dataset["qid"] == qid]
    assert len(df) != 0
    
    if len(df.columns) <= 200:
        vector_size = 128
    elif 200 < len(df.columns) <= 300:
        vector_size = 256
    else:
        vector_size = 768

    # valid_columns = [str(x) for x in range(1,vector_size+1)] + ["relevance", "bias", "nfair"]
    valid_columns = [str(x) for x in range(1,vector_size+1)] + ["relevance"]
    df = df[valid_columns]
    # print(f"query featues: {df.columns}")
    return df.values


def get_model_inputs(state, action, dataset, verbose=True) -> np.ndarray:

    temp = [state.t] + get_features(state.qid, action, dataset)
    temp = [float(x) for x in temp]

    temp_array = np.array(temp)
    
    min_val = temp_array.min()
    max_val = temp_array.max()
    if max_val - min_val > 0:  # Avoid division by zero
        normalized_temp = (temp_array - min_val) / (max_val - min_val)
    else:
        normalized_temp = temp_array

    return np.array(normalized_temp)

def get_multiple_model_inputs(state, doc_list, dataset) -> np.ndarray:

    return np.insert(get_query_features(state.qid, doc_list, dataset), 0, state.t, axis=1)





