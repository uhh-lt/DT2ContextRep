import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from torch_geometric.data import DataLoader as GraphDataLoader
from transformers import AutoTokenizer

import config
import utils
from data_loader import WrapperDataset, Seid_Amharic_SentimentLMDataset, Amharic_EvalAuxDataset
from model import SentenceLevelNet

def get_sentence_embeds(loader):
    embeds_list = list()
    sentences_list = list()
    for sentence_batch in loader:
        [[input_ids, token_type_ids, attention_mask], batch_aux_1] = sentence_batch
        latent, y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1)
        embeds_list.append(np.array(latent.squeeze(0).detach().cpu()))   # enforce batch_size = 1
        sentences_list.append(batch_aux_1.sentence)
    return np.array(embeds_list), sentences_list

def tsne_plot(embeddings, sentences, output_plot_save_path, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], n_components=2, random_state=42):
    assert(embeddings.shape[0] == len(sentences))
    assert(embeddings.shape[0] <= len(colors))
    
    tsne_model = TSNE(perplexity=42, n_components=2, random_state=random_state, n_jobs=-1)
    values = tsne_model.fit_transform(embeddings)
    assert(embeddings.shape[0] == values.shape[0])
    
    dpi = 300
    plt.figure(figsize=(8, 6), edgecolor='black', linewidth=2, dpi=dpi)
    
    for i in range(embeddings.shape[0]):
        plt.scatter(values[i][0], values[i][1], c=colors[i], label=sentences[i])
    
    plt.savefig(output_plot_save_path, dpi=dpi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_plot_save_path', required=True)
    args = parser.parse_args()

    assert(config.debug == True)
    assert(config.train == False)
    assert(config.batch_size == 1)
    assert(config.experiment == 'seid_amharic_cleaned_sentiment')

    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.Seid_Amharic_Sentiment_cleaned_labels))
    model.to(config.device)

    data_frame = utils.get_Seid_amharic_sentiment_cleaned_data_frame(config.Amharic_eval_path)
    LM_dataset = Seid_Amharic_SentimentLMDataset(data_frame, tokenizer)
    dataset_aux = Amharic_EvalAuxDataset(root='../data/sentence_embed_eval/', data_frame=data_frame, DT_G=DT_G, is_sentence_1=True)
    loader = GraphDataLoader(WrapperDataset(LM_dataset, dataset_aux), batch_size=config.batch_size)
    sentence_embeds, sentences_list = get_sentence_embeds(loader)
    
    tsne_plot(sentence_embeds, sentences_list, args.output_plot_save_path)