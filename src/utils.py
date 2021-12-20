import argparse
import math
import json

from tqdm import tqdm
from nltk.tag import pos_tag
import pandas as pd 
import networkx as nx
import torch

import config

def get_relevant_tokens(word_count_path, threshold):
    d = pd.read_csv(word_count_path, sep='\t', header=None, quotechar=None, quoting=3)
    d.columns = ['token', 'count']
    d = d.loc[d['count'] > threshold]
    return d.token.tolist()

def prune_dt(input_dt_edges_path, relevant_tokens, output_dt_edges_path):
    d = pd.read_csv(input_dt_edges_path, sep='\t', header=None, quotechar=None, quoting=3)
    d.columns = ['word1', 'word2', 'weight']
    d = d.loc[d['word1'].isin(relevant_tokens) & d['word2'].isin(relevant_tokens)]
    d.to_csv(output_dt_edges_path, sep='\t', index=False, header=None, quotechar=None, quoting=3)

def update_POS_tags(input_DT_path, output_DT_path):
    d = pd.read_csv(input_DT_path, sep='\t', header=None, quotechar=None, quoting=3)
    d.columns = ['word1', 'word2', 'weight']
    def replace_POS(e):
        # https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
        d = {'NP': 'NNP', 'NPS': 'NNPS', 'PP': 'PRP', 'PP$': 'PRP$'}
        word, pos = e.rsplit(config.DT_token_pos_delimiter, 1)
        if(pos in d.keys()):
            return f'{word}{config.DT_token_pos_delimiter}{d[pos]}'
        else:
            return f'{word}{config.DT_token_pos_delimiter}{pos}'
    d.word1 = d.word1.apply(lambda x: replace_POS(x))
    d.word2 = d.word2.apply(lambda x: replace_POS(x))
    d.to_csv(output_DT_path, sep='\t', index=False, header=None, quotechar=None, quoting=3)

def load_DT(DT_edges_path=config.prune_DT_edges_path):
    df = pd.read_csv(DT_edges_path, header=None, sep='\t', quotechar=None, quoting=3)
    df.columns = ['word1', 'word2', 'weight']
    G = nx.from_pandas_edgelist(df, 'word1', 'word2', 'weight')
    print('Loaded the DT networkx graph')
    return G

def edge_weight_u_v(DT_G, node1, node2):
    try:
        # ensure that shortest path over self-loops are not computed
        shortest_path_length = nx.algorithms.shortest_paths.generic.shortest_path_length(G=DT_G, source=node1, target=node2, weight=None)
        score = math.exp((-1) * (config.path_lambda) * (shortest_path_length - 1))
        path_exists = True
    except nx.exception.NodeNotFound:
        score = -1
        path_exists = False
    except nx.exception.NetworkXNoPath:
        score = -1
        path_exists = False
    return path_exists, score

def setup_graph_edges(DT_G, sentence):
    space_tokenized_sentence = sentence.split()
    if(config.is_en):
        pos_tagged_space_tokenized_sentence = [token + config.DT_token_pos_delimiter + tag for (token, tag) in pos_tag(space_tokenized_sentence)]
    else:
        # no POS tagger used in the non english DT
        pos_tagged_space_tokenized_sentence = space_tokenized_sentence
    assert(len(pos_tagged_space_tokenized_sentence) == len(space_tokenized_sentence))
    
    # to ensure that every graph has edges - setup the mandatory self-loops
    _edge_index = [[i, i] for i in range(len(space_tokenized_sentence))]
    _edge_attr = [[1] for _ in _edge_index]

    for i in range(len(space_tokenized_sentence)):
        for j in range(i+1, len(space_tokenized_sentence)):
            assert(i != j)
            path_exists, edge_weight = edge_weight_u_v(DT_G, pos_tagged_space_tokenized_sentence[i], pos_tagged_space_tokenized_sentence[j])
            if(path_exists):
                _edge_index.append([i, j])
                _edge_attr.append([edge_weight])
                _edge_index.append([j, i])
                _edge_attr.append([edge_weight])
    edge_index = torch.LongTensor(_edge_index).to(config.device)
    edge_index = torch.transpose(edge_index, 0, 1)
    # shape(edge_index) = [2, num_edges]
    edge_attr = torch.FloatTensor(_edge_attr).to(config.device)
    # shape(edge_attr) = [num_edges, 1]

    return edge_index, edge_attr

def get_sentences_encoded_dict(tokenizer, sentences, max_length):
    assert(len(sentences) == 1 or len(sentences) == 2)
    
    if(len(sentences) == 1):
        encoded_dict = tokenizer.encode_plus(sentences[0], add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
    elif(len(sentences) == 2):
        encoded_dict = tokenizer.encode_plus(sentences[0], sentences[1], add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
    
    input_ids = encoded_dict['input_ids'][0].to(config.device)
    if(config.lm_model_name.startswith('roberta') or config.lm_model_name.startswith('xlm-roberta')):
        token_type_ids = torch.zeros_like(input_ids)
    else:
        token_type_ids = encoded_dict['token_type_ids'][0].to(config.device)
    attention_mask = encoded_dict['attention_mask'][0].to(config.device)
    
    return input_ids, token_type_ids, attention_mask

def get_label_embedding(label, label_dict):
    assert(label in label_dict)
    vec = torch.zeros(len(label_dict), dtype=torch.float, device=config.device)
    vec[label_dict[label]] = 1
    vec = torch.unsqueeze(vec, 0)
    # shape(vec) = [1, len(label_dict)]
    return vec

def get_score_embedding(score):
    vec = torch.tensor([score], dtype=torch.float).unsqueeze(0).to(config.device)
    # shape(vec) = [1, 1]
    return vec

def get_WiC_data_frame(WiC_data_path, WiC_gold_path):
    df_data = pd.read_csv(WiC_data_path, header=None, sep='\t')
    df_gold = pd.read_csv(WiC_gold_path, header=None, sep='\t')
    df = pd.concat([df_data, df_gold], axis=1, sort=False)
    df.columns = ['target', 'pos', 'indices', 'context_1', 'context_2', 'label']
    print('Loaded the WiC dataset')
    return df

def get_RTE_data_frame(RTE_data):
    df = pd.read_csv(RTE_data, sep='\t')
    print(f'RTE dataframe loaded from {RTE_data}')
    return df

def get_STS_B_data_frame(STS_B_data, columns=config.STS_B_columns):
    # Loader adapted from https://colab.research.google.com/github/hybridnlp/tutorial/blob/master/07a_semantic_claim_search.ipynb
    rows = []
    print(f'Loading STS_B dataset from {STS_B_data}')
    with open(STS_B_data, mode='r', encoding='utf') as f:
        lines = f.readlines()
        print(f'Reading {len(lines)} lines from {STS_B_data}')
        for lnr, line in enumerate(lines):
            cols = line.split('\t')
            assert len(cols) >= 7, 'line %s has %s columns instead of %s:\n\t%s' % (
                lnr, len(cols), 7, "\n\t".join(cols)
            ) 
            cols = cols[:7]
            assert len(cols) == 7
            rows.append(cols)
    df = pd.DataFrame(rows, columns=columns)
    df.sent_a = df.sent_a.str.strip()
    df.sent_b = df.sent_b.str.strip()
    # score is read as a string, so add a copy with correct type
    df['score_f'] = df['score'].astype('float64')
    return df

def get_MRPC_data_frame(MRPC_data, columns=config.MRPC_columns):
    # Loader adapted from https://colab.research.google.com/github/hybridnlp/tutorial/blob/master/07a_semantic_claim_search.ipynb
    rows = []
    print(f'Loading MRPC dataset from {MRPC_data}')
    with open(MRPC_data, mode='r', encoding='utf') as f:
        lines = f.readlines()[1:]
        print(f'Reading {len(lines)} lines from {MRPC_data}')
        for lnr, line in enumerate(lines):
            cols = line.split('\t')
            assert len(cols) == 5
            rows.append(cols)
    df = pd.DataFrame(rows, columns=columns)
    return df

def get_SST_2_data_frame(SST_2_data):
    d = pd.read_csv(SST_2_data, sep='\t')
    return d

def get_CoLA_data_frame(CoLA_data):
    d = pd.read_csv(CoLA_data, sep='\t', header=None)
    d.columns = ['category', 'label', 'domain', 'sentence']
    return d

def get_WNLI_translated_data_frame(WNLI_data):
    d = pd.read_csv(WNLI_data)
    return d

def get_IITP_product_reviews_data_frame(ITP_product_reviews_data):
    d = pd.read_csv(ITP_product_reviews_data, header=None)
    d.columns = ['label', 'sentence']
    d = d.convert_dtypes()
    return d

def get_MIDAS_discourse_json(json_path):
    json_file = open(json_path)
    orig_d = json.load(json_file)
    prune_d = list()
    for d in orig_d:
        if(d['Discourse Mode'] in config.MIDAS_discourse_labels.keys()):
            prune_d.append(d)
    return prune_d

def get_DPIL_data_frame(DPIL_data):
    d = pd.read_csv(DPIL_data, header=None)
    d.columns = ['sentence_1', 'sentence_2', 'label']
    return d

def get_KhondokerIslam_bengali_data_frame(KhondokerIslam_bengali_path):
    d = pd.read_csv(KhondokerIslam_bengali_path)
    return d

def get_rezacsedu_sentiment_data_frame(rezacsedu_sentiment_path):
    d = pd.read_csv(rezacsedu_sentiment_path, header=None)
    d.columns = ['text', 'label']
    return d

def get_BEmoC_data_frame(BEmoC_path):
    d = pd.read_excel(BEmoC_path)
    return d

def get_Seid_amharic_sentiment_data_frame(Seid_amharic_sentiment_path):
    d = pd.read_csv(Seid_amharic_sentiment_path, header=None)
    d.columns = ['tmp']
    d[['label', 'sentence']] = d.tmp.str.split(' ', 1, expand=True)
    return d

def get_Seid_amharic_sentiment_cleaned_data_frame(Seid_amharic_sentiment_path):
    d = pd.read_csv(Seid_amharic_sentiment_path, header=None)
    d.columns = ['tmp']
    d[['label', 'sentence']] = d.tmp.str.split(' ', 1, expand=True)
    result_d = d.loc[d.label.isin(list(config.Seid_Amharic_Sentiment_cleaned_labels.keys()))]
    result_d = result_d.reset_index()
    return result_d

def get_germeval2018_data_frame(germeval2018_path):
    d = pd.read_csv(germeval2018_path, sep='\t', header=None, quotechar=None, quoting=3)
    d.columns = ['sentence', 'subtask_1_label', 'subtask_2_label']
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prune_DT', action='store_true')
    group.add_argument('--update_POS_tags', action='store_true')
    args = parser.parse_args()
    if(args.prune_DT):
        threshold = config.prune_DT_frequency_threshold
        print(f'Threshold frequency: {threshold}')
        relevant_tokens = get_relevant_tokens(config.DT_word_count_path, threshold)
        print('Compiled relevant_tokens')
        prune_dt(config.DT_edges_path, relevant_tokens, config.prune_DT_edges_path)
        print('Saved pruned DT')
    elif(args.update_POS_tags):
        assert(config.update_POS)
        update_POS_tags(input_DT_path=config.prune_DT_edges_path_orig, output_DT_path=config.prune_DT_edges_path)
