import torch 
import torch.nn as nn
from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.nn import GCNConv, LEConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn.util import masked_softmax, masked_mean

import config
import utils
from data_loader import WrapperDataset, WiCLMDataset, WiCAuxDataset


class LMNet(nn.Module):
    def __init__(self, tokenizer):
        super(LMNet, self).__init__()
        self.tokenizer = tokenizer
        self.lm_model = AutoModel.from_pretrained(config.lm_model_name)
        self.lm_model.train()
        if(config.lm_model_name.startswith('bert') or config.lm_model_name.startswith('xlm-roberta') or config.lm_model_name.startswith('roberta')):
            self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2=None):
        last_hidden_state, pooler_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if(config.lm_model_name.startswith('bert') or config.lm_model_name.startswith('xlm-roberta') or config.lm_model_name.startswith('roberta')):
            last_hidden_state = self.dropout(last_hidden_state)
            pooler_output = self.dropout(pooler_output)

        x_1 = torch.zeros_like(last_hidden_state)
        sentence_mask_1 = torch.zeros_like(attention_mask)

        if(batch_aux_2):
            x_2 = torch.zeros_like(last_hidden_state)
            sentence_mask_2 = torch.zeros_like(attention_mask)

        # iterate over the batch of inputs
        for i in range(input_ids.shape[0]):
            input_id = input_ids[i]
            subword_tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            
            sentence_1 = batch_aux_1.sentence[i]
            sentence_1_space_split = sentence_1.split()
            # get sub-word starting index of first sentence
            sentence_1_index = subword_tokens.index(self.tokenizer.cls_token) + 1
            # list of word embeddings (computed by averaging subwords)
            sentence_1_x = list()
            for word_index, word in enumerate(sentence_1_space_split):
                if(config.lm_model_name.startswith('roberta') and (word_index != 0)):
                    # this was noted only in the tokenizer of monolingual english roberta
                    word = ' ' + word
                num_subword_tokens = len(self.tokenizer.tokenize(word))
                x = torch.mean(last_hidden_state[i][sentence_1_index: sentence_1_index+num_subword_tokens], dim=0)
                assert(torch.isnan(x).any() == False)
                sentence_1_index = sentence_1_index + num_subword_tokens
                sentence_1_x.append(x)
            assert(len(sentence_1_x) == len(sentence_1_space_split))
            x_1[i, :len(sentence_1_x), :] = torch.stack(sentence_1_x)
            sentence_mask_1[i, :len(sentence_1_x)] = torch.ones(len(sentence_1_x))

            if(batch_aux_2):
                sentence_2 = batch_aux_2.sentence[i]
                sentence_2_space_split = sentence_2.split()
                # get sub-word starting index of second sentence
                if(config.lm_model_name.startswith('roberta') or config.lm_model_name.startswith('xlm-roberta')):
                    sentence_2_index = subword_tokens.index(self.tokenizer.sep_token) + 2
                else:    
                    sentence_2_index = subword_tokens.index(self.tokenizer.sep_token) + 1
                # list of word embeddings (computed by averaging subwords)
                sentence_2_x = list()
                for word_index, word in enumerate(sentence_2_space_split):
                    if(config.lm_model_name.startswith('roberta') and (word_index != 0)):
                        # this was noted only in the tokenizer of monolingual english roberta
                        word = ' ' + word
                    num_subword_tokens = len(self.tokenizer.tokenize(word))
                    x = torch.mean(last_hidden_state[i][sentence_2_index: sentence_2_index+num_subword_tokens], dim=0)
                    assert(torch.isnan(x).any() == False)
                    sentence_2_index = sentence_2_index + num_subword_tokens
                    sentence_2_x.append(x)
                assert(len(sentence_2_x) == len(sentence_2_space_split))
                x_2[i, :len(sentence_2_x), :] = torch.stack(sentence_2_x)
                sentence_mask_2[i, :len(sentence_2_x)] = torch.ones(len(sentence_2_x))
        
        if(batch_aux_2):
            return pooler_output, x_1, sentence_mask_1, x_2, sentence_mask_2
        else:
            return pooler_output, x_1, sentence_mask_1


class GNNNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dropout)

        if(config.use_gcn):
            self.conv_1 = GCNConv(in_channels=self.in_dim, out_channels=self.in_dim, node_dim=1)
            self.conv_2 = GCNConv(in_channels=self.in_dim, out_channels=self.in_dim, node_dim=1)
        elif(config.use_le):
            self.conv_1 = LEConv(in_channels=self.in_dim, out_channels=self.in_dim, node_dim=1)
            self.conv_2 = LEConv(in_channels=self.in_dim, out_channels=self.out_dim, node_dim=1)

    def forward(self, x, batch_aux):
        edge_index, edge_attr = batch_aux.edge_index, batch_aux.edge_attr
        hiddens = x
        # edge_attr has to be flattned into edge_weight: https://github.com/rusty1s/pytorch_geometric/issues/1441
        edge_attr = edge_attr.flatten()
        if(config.use_gcn or config.use_le):
            hiddens = self.dropout(self.relu(self.conv_1(hiddens, edge_index, edge_weight=edge_attr)))
            hiddens = self.dropout(self.relu(self.conv_2(hiddens, edge_index, edge_weight=edge_attr)))
        return hiddens


class InteractGraphsNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InteractGraphsNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.relu = nn.ReLU()

        self.attn_biaffine = BilinearMatrixAttention(self.in_dim, self.in_dim, use_input_biases=True)
        self.attn_proj = nn.Linear(4*self.in_dim, self.out_dim)

    def forward(self, h_1, sentence_mask_1, h_2, sentence_mask_2, batch_aux_1, batch_aux_2):
        attn = self.attn_biaffine(h_1, h_2)
        
        # normalized_attn_1.shape = [batch_size, seq_len_1, seq_len_2]
        normalized_attn_1 = masked_softmax(attn, sentence_mask_1.unsqueeze(2), dim=1)
        attended_1 = normalized_attn_1.transpose(1, 2).bmm(h_1)
        new_h_2 = torch.cat([h_2, attended_1, h_2-attended_1, h_2*attended_1], dim=-1)
        new_h_2 = self.relu(self.attn_proj(new_h_2))

        # normalized_attn_2.shape = [batch_size, seq_len_1, seq_len_2]
        normalized_attn_2 = masked_softmax(attn, sentence_mask_2.unsqueeze(2), dim=1)
        attended_2 = normalized_attn_2.bmm(h_2)
        new_h_1 = torch.cat([h_1, attended_2, h_1-attended_2, h_1*attended_2], dim=-1)
        new_h_1 = self.relu(self.attn_proj(new_h_1))

        return new_h_1, new_h_2
        

class Net(nn.Module):
    def __init__(self, tokenizer):
        super(Net, self).__init__()
        self.tokenizer = tokenizer

        self.lm_net = LMNet(tokenizer=self.tokenizer)
        
        self.in_dim = self.lm_net.lm_model.config.hidden_size
        self.out_dim = self.lm_net.lm_model.config.hidden_size

        if(config.use_gcn or config.use_le):
            self.pre_interact_gnn = GNNNet(in_dim=self.in_dim, out_dim=self.in_dim)

            if(config.is_sentence_pair_task):
                self.interact_graphs_net = InteractGraphsNet(in_dim=self.in_dim, out_dim=self.in_dim)
                self.post_interact_gnn = GNNNet(in_dim=self.in_dim, out_dim=self.out_dim)
            
    def forward(self, input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2=None):
        if(config.is_sentence_pair_task):
            lm_pooler_output, x_1, sentence_mask_1, x_2, sentence_mask_2 = self.lm_net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)

            if(not (config.use_gcn or config.use_le)):
                return lm_pooler_output, sentence_mask_1, sentence_mask_2, x_1, x_2

            else:
                h_1 = self.pre_interact_gnn(x_1, batch_aux_1)
                h_2 = self.pre_interact_gnn(x_2, batch_aux_2)
                new_h_1, new_h_2 = self.interact_graphs_net(h_1, sentence_mask_1, h_2, sentence_mask_2, batch_aux_1, batch_aux_2)
                
                post_interact_gnn_h_1 = self.post_interact_gnn(new_h_1, batch_aux_1)
                post_interact_gnn_h_2 = self.post_interact_gnn(new_h_2, batch_aux_2)
                return lm_pooler_output, sentence_mask_1, sentence_mask_2, x_1, x_2, post_interact_gnn_h_1, post_interact_gnn_h_2
        
        else:
        
            lm_pooler_output, x_1, sentence_mask_1 = self.lm_net(input_ids, token_type_ids, attention_mask, batch_aux_1)
            
            if(not (config.use_gcn or config.use_le)):
                return lm_pooler_output, sentence_mask_1, x_1
            
            else:
                h_1 = self.pre_interact_gnn(x_1, batch_aux_1)
                return lm_pooler_output, sentence_mask_1, x_1, h_1


class WordLevelNet(nn.Module):
    def __init__(self, tokenizer, num_output_classes):
        super(WordLevelNet, self).__init__()
        self.tokenizer = tokenizer
        self.num_output_classes = num_output_classes
        self.net = Net(tokenizer=self.tokenizer)

        if(config.is_sentence_pair_task):
            if(not (config.use_gcn or config.use_le)):
                self.output_probs_linear = nn.Linear(self.net.out_dim, self.num_output_classes)    
            else:
                self.layer_norm = nn.LayerNorm(3*self.net.out_dim)
                self.output_probs_linear = nn.Linear(3*self.net.out_dim, self.num_output_classes)
        else:
            if(not (config.use_gcn or config.use_le)):
                self.output_probs_linear = nn.Linear(self.net.out_dim, self.num_output_classes)
            else:
                self.layer_norm = nn.LayerNorm(2*self.net.out_dim)
                self.output_probs_linear = nn.Linear(2*self.net.out_dim, self.num_output_classes)
    
    def forward(self, input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2=None):
        if(config.is_sentence_pair_task):
            assert(batch_aux_2)

            if(not (config.use_gcn or config.use_le)):
                lm_pooler_output, _, _, x_1, x_2 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
                
                concat_latent = lm_pooler_output
                
            else:
                lm_pooler_output, _, _, x_1, x_2, post_interact_gnn_h_1, post_interact_gnn_h_2 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)

                batch_indices_1 = batch_aux_1.word_loc.unsqueeze(1).repeat(1, self.net.out_dim).unsqueeze(1)
                post_interact_gnn_word_embed_1 = torch.gather(post_interact_gnn_h_1, 1, batch_indices_1).squeeze(1)

                batch_indices_2 = batch_aux_2.word_loc.unsqueeze(1).repeat(1, self.net.out_dim).unsqueeze(1)
                post_interact_gnn_word_embed_2 = torch.gather(post_interact_gnn_h_2, 1, batch_indices_2).squeeze(1)

                concat_latent = self.layer_norm(torch.cat([lm_pooler_output, post_interact_gnn_word_embed_1, post_interact_gnn_word_embed_2], dim=1))
                
        else:

            if(not (config.use_gcn or config.use_le)):
                lm_pooler_output, _, x_1 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1)

                concat_latent = lm_pooler_output
            
            else:
                lm_pooler_output, _, x_1, h_1 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1)
                
                batch_indices_1 = batch_aux_1.word_loc.unsqueeze(1).repeat(1, self.net.out_dim).unsqueeze(1)
                h_1_word_embed_1 = torch.gather(h_1, 1, batch_indices_1).squeeze(1)

                concat_latent = self.layer_norm(torch.cat([lm_pooler_output, h_1_word_embed_1], dim=1))
        
        output_probs = self.output_probs_linear(concat_latent)
        if(config.debug == True):
            return concat_latent, output_probs
        elif(config.debug == False):
            return output_probs


class SentenceLevelNet(nn.Module):
    def __init__(self, tokenizer, num_output_classes):
        super(SentenceLevelNet, self).__init__()
        self.tokenizer = tokenizer
        self.num_output_classes = num_output_classes
        self.net = Net(tokenizer=self.tokenizer)

        if(config.is_sentence_pair_task):
            if(not (config.use_gcn or config.use_le)):
                self.output_probs_linear = nn.Linear(self.net.out_dim, self.num_output_classes)
            else:
                self.layer_norm = nn.LayerNorm(3*self.net.out_dim)
                self.output_probs_linear = nn.Linear(3*self.net.out_dim, self.num_output_classes)
        else:
            if(not (config.use_gcn or config.use_le)):
                self.output_probs_linear = nn.Linear(self.net.out_dim, self.num_output_classes)
            else:
                self.layer_norm = nn.LayerNorm(2*self.net.out_dim)
                self.output_probs_linear = nn.Linear(2*self.net.out_dim, self.num_output_classes)
        
    def forward(self, input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2=None):
        if(config.is_sentence_pair_task):
            assert(batch_aux_2)

            if(not (config.use_gcn or config.use_le)):
                lm_pooler_output, sentence_mask_1, sentence_mask_2, x_1, x_2 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)

                concat_latent = lm_pooler_output

            else:
                lm_pooler_output, sentence_mask_1, sentence_mask_2, x_1, x_2, post_interact_gnn_h_1, post_interact_gnn_h_2 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)

                mean_post_interact_gnn_h_1 = masked_mean(post_interact_gnn_h_1, sentence_mask_1.unsqueeze(-1), 1)
                mean_post_interact_gnn_h_2 = masked_mean(post_interact_gnn_h_2, sentence_mask_2.unsqueeze(-1), 1)

                concat_latent = self.layer_norm(torch.cat([lm_pooler_output, mean_post_interact_gnn_h_1, mean_post_interact_gnn_h_2], dim=1))
        
        else:
            
            if(not (config.use_gcn or config.use_le)):
                lm_pooler_output, sentence_mask_1, x_1 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1)

                concat_latent = lm_pooler_output
            
            else:
                lm_pooler_output, sentence_mask_1, x_1, h_1 = self.net(input_ids, token_type_ids, attention_mask, batch_aux_1)

                mean_h_1 = masked_mean(h_1, sentence_mask_1.unsqueeze(-1), 1)

                concat_latent = self.layer_norm(torch.cat([lm_pooler_output, mean_h_1], dim=1))
    
        output_probs = self.output_probs_linear(concat_latent)
        if(config.debug == True):
            return concat_latent, output_probs
        elif(config.debug == False):
            return output_probs


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()

    WiC_train_data_frame = utils.get_WiC_data_frame(config.WiC_train_data_path, config.WiC_train_gold_path)
    LM_dataset = WiCLMDataset(WiC_train_data_frame, tokenizer)
    train_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
    train_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_2=True)
    train_loader = GraphDataLoader(WrapperDataset(LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)
    batch = next(iter(train_loader))

    WiC_dev_data_frame = utils.get_WiC_data_frame(config.WiC_dev_data_path, config.WiC_dev_gold_path)
    LM_dataset = WiCLMDataset(WiC_dev_data_frame, tokenizer)
    dev_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
    dev_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
    dev_loader = GraphDataLoader(WrapperDataset(LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)
    batch = next(iter(dev_loader))
    print(f'batch: {batch}')
    [[input_ids, token_type_ids, attention_mask], batch_aux_1, batch_aux_2] = batch

    wic_word_level_net = WordLevelNet(tokenizer=tokenizer, num_output_classes=len(config.WiC_labels))
    wic_word_level_net.to(config.device)
    wic_word_level_net(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
