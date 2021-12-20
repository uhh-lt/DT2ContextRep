import os
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import DataLoader as GraphDataLoader
from transformers import AutoTokenizer

import utils
import config


class WrapperDataset(Dataset):
    def __init__(self, *datasets):
        assert(all(len(dataset) == len(datasets[0]) for dataset in datasets))
        self.datasets = list()
        for dataset in datasets:
            self.datasets.append(dataset)
        assert(len(self.datasets) > 0)
    
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        output = list()
        for dataset in self.datasets:
            output.append(dataset[idx])
        return tuple(output)


class WiCLMDataset(Dataset):
    max_length = 80
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)

    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on WiC dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['context_1'].lower()
            sentence_2 = row['context_2'].lower()
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on WiC dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['context_1'].lower()
        sentence_2 = row['context_2'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class WiCAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(WiCAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed WiC Aux dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            index_1, index_2 = list(map(int, row['indices'].split('-')))
            if(self.is_sentence_1):
                context = row['context_1']
                word_loc = index_1
            elif(self.is_sentence_2):
                context = row['context_2']
                word_loc = index_2
            context = context.lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.WiC_labels).to(config.device)
            data['word_loc'] = torch.LongTensor([word_loc]).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class RTELMDataset(Dataset):
    max_length = 307
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on RTE dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence1'].lower()
            sentence_2 = row['sentence2'].lower()
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on RTE dataset: {max_length}')
    
    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence1'].lower()
        sentence_2 = row['sentence2'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class RTEAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(RTEAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed RTE dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            if(self.is_sentence_1):
                context = row['sentence1']
            elif(self.is_sentence_2):
                context = row['sentence2']
            context = context.lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.RTE_labels).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class STS_BLMDataset(Dataset):
    max_length = 127
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on STS_B dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sent_a'].lower()
            sentence_2 = row['sent_b'].lower()
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on STS_B dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sent_a'].lower()
        sentence_2 = row['sent_b'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class STS_BAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(STS_BAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed STS_B dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            score = row['score_f']
            if(self.is_sentence_1):
                context = row['sent_a']
            elif(self.is_sentence_2):
                context = row['sent_b']
            context = context.lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_score_embedding(score)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class MRPCLMDataset(Dataset):
    max_length = 139
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
        
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on MRPC dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['#1 String'].lower()
            sentence_2 = row['#2 String'].lower()
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on MRPC dataset: {max_length}')
    
    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['#1 String'].lower()
        sentence_2 = row['#2 String'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class MRPCAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(MRPCAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed MRPC dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['Quality']
            if(self.is_sentence_1):
                context = row['#1 String']
            elif(self.is_sentence_2):
                context = row['#2 String']
            context = context.lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.MRPC_labels).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class SST_2LMDataset(Dataset):
    max_length = 81
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on SST_2 dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence'].lower()
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on SST_2 dataset: {max_length}')
    
    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class SST_2AuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(SST_2AuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed SST_2 dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence'].lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.SST_2_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class CoLA_LMDataset(Dataset):
    max_length = 48
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on SST_2 dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence'].lower()
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on SST_2 dataset: {max_length}')
    
    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence'].lower()
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class CoLAAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(CoLAAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed CoLA dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence'].lower()
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.CoLA_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class WNLI_TranslatedLMDataset(Dataset):
    max_length = 171
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on WLNI_Translated dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence1']
            sentence_2 = row['sentence2']
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on WLNI_Translated dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence1']
        sentence_2 = row['sentence2']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask

class WNLI_TranslatedAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(WNLI_TranslatedAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed WNLI_Translated Aux dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            if(self.is_sentence_1):
                context = row['sentence1']
            elif(self.is_sentence_2):
                context = row['sentence2']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.WNLI_translated_labels).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class IITP_Product_ReviewsLMDataset(Dataset):
    max_length = 179
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on IITP_Product_Reviews dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on IITP_Product_Reviews dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class IITP_Product_ReviewsAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(IITP_Product_ReviewsAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed IITP_Product_Reviews dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.IITP_product_reviews_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class MIDAS_DiscourseLMDataset(Dataset):
    max_length = 321
    def __init__(self, json_data, tokenizer):
        self.data = json_data
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)

    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on MIDAS_DiscourseLMDataset dataset')
        max_length = 0
        for row_index, row in tqdm(enumerate(self.data), total=len(self.data)):
            sentence_1 = row['Sentence']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on MIDAS_DiscourseLMDataset dataset: {max_length}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        sentence_1 = row['Sentence']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class MIDAS_DiscourseAuxDataset(GraphDataset):
    def __init__(self, root, json_data, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data = json_data
        self.DT_G = DT_G
        super(MIDAS_DiscourseAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed MIDAS_Discourse dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(len(self.data))]
    
    def process(self):
        for row_index, row in tqdm(enumerate(self.data), total=len(self.data)):
            label = row['Discourse Mode']
            context = row['Sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.MIDAS_discourse_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class DPIL_Subtask_1LMDataset(Dataset):
    max_length = 203
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on DPIL_Subtask_1 dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence_1']
            sentence_2 = row['sentence_2']
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on DPIL_Subtask_1 dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence_1']
        sentence_2 = row['sentence_2']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class DPIL_Subtask_1AuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(DPIL_Subtask_1AuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed DPIL_Subtask_1 Aux dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            if(self.is_sentence_1):
                context = row['sentence_1']
            elif(self.is_sentence_2):
                context = row['sentence_2']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.DPIL_subtask_1_labels).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class DPIL_Subtask_2LMDataset(Dataset):
    max_length = 210
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on DPIL_Subtask_2 dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence_1']
            sentence_2 = row['sentence_2']
            input_ids = self.tokenizer.encode(sentence_1, sentence_2, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on DPIL_Subtask_2 dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence_1']
        sentence_2 = row['sentence_2']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1, sentence_2], self.max_length)
        return input_ids, token_type_ids, attention_mask


class DPIL_Subtask_2AuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        self.data_frame = data_frame
        self.DT_G = DT_G
        assert(sum([is_sentence_1, is_sentence_2]) == 1)
        self.is_sentence_1 = is_sentence_1
        self.is_sentence_2 = is_sentence_2
        super(DPIL_Subtask_2AuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed DPIL_Subtask_2 Aux dataset from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if(self.is_sentence_1):
            return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
        elif(self.is_sentence_2):
            return [f'aux_2_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            if(self.is_sentence_1):
                context = row['sentence_1']
            elif(self.is_sentence_2):
                context = row['sentence_2']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.DPIL_subtask_2_labels).to(config.device)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            if(self.is_sentence_1):
                torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))
            elif(self.is_sentence_2):
                torch.save(data, os.path.join(self.processed_dir, f'aux_2_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if(self.is_sentence_1):
            data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        elif(self.is_sentence_2):
            data = torch.load(os.path.join(self.processed_dir, f'aux_2_{idx}.pt'))
        return data


class KhondokerIslam_BengaliLMDataset(Dataset):
    max_length = 218
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on KhondokerIslam_Bengali dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['Data']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on KhondokerIslam_Bengali dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['Data']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask

class KhondokerIslam_BengaliAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(KhondokerIslam_BengaliAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed KhondokerIslam_Bengali from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['Sentiment']
            context = row['Data']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.KhondokerIslam_bengali_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Rezacsedu_SentimentLMDataset(Dataset):
    max_length = 260
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on Rezacsedu_Sentiment dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['text']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on Rezacsedu_Sentiment dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['text']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask

class Rezacsedu_SentimentAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Rezacsedu_SentimentAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Rezacsedu_Sentiment from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['text']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.rezacsedu_sentiment_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class BEmoCLMDataset(Dataset):
    max_length = 460
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on BEmoCLMDataset dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['cleaned']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on BEmoCLMDataset dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['cleaned']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask

class BEmoCLMDatasetAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(BEmoCLMDatasetAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed BEmoCL from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['classes']
            context = row['cleaned']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.BEmoC_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Seid_Amharic_SentimentLMDataset(Dataset):
    max_length = 95
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on Seid_Amharic_Sentiment dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on Seid_Amharic_Sentiment dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class Seid_Amharic_SentimentAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Seid_Amharic_SentimentAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Seid_Amharic_Sentiment from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.Seid_Amharic_Sentiment_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Seid_Amharic_Cleaned_SentimentAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Seid_Amharic_Cleaned_SentimentAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Seid_Amharic_Cleaned_Sentiment from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.Seid_Amharic_Sentiment_cleaned_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Amharic_EvalAuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Amharic_EvalAuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Amharic_Eval from {root}')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.Amharic_eval_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Germeval2018LMDataset(Dataset):
    max_length = 228
    def __init__(self, data_frame, tokenizer):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        assert(self.max_length <= self.tokenizer.model_max_length)
    
    def _compute_max_length(self):
        print(f'Computing max length (sub-word + special tokens) on Germeval2018 dataset')
        max_length = 0
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            sentence_1 = row['sentence']
            input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
        max_length = max_length if max_length <= self.tokenizer.model_max_length else self.tokenizer.model_max_length
        print(f'Max length (sub-word + special tokens) on Germeval2018 dataset: {max_length}')

    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx, :]
        sentence_1 = row['sentence']
        input_ids, token_type_ids, attention_mask = utils.get_sentences_encoded_dict(self.tokenizer, [sentence_1], self.max_length)
        return input_ids, token_type_ids, attention_mask


class Germeval2018_Subtask_1AuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Germeval2018_Subtask_1AuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Germeval_Subtask_1 from {root}')

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['subtask_1_label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.germeval2018_subtask_1_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


class Germeval2018_Subtask_2AuxDataset(GraphDataset):
    def __init__(self, root, data_frame, DT_G, is_sentence_1=False, is_sentence_2=False, transform=None, pre_transform=None):
        assert(is_sentence_1 == True and is_sentence_2 == False)
        self.data_frame = data_frame
        self.DT_G = DT_G
        super(Germeval2018_Subtask_2AuxDataset, self).__init__(root, transform, pre_transform)
        print(f'Loaded processed Germeval_Subtask_2 from {root}')

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'aux_1_{i}.pt' for i in range(self.data_frame.shape[0])]
    
    def process(self):
        for row_index, row in tqdm(self.data_frame.iterrows(), total=self.data_frame.shape[0]):
            label = row['subtask_2_label']
            context = row['sentence']
            edge_index, edge_attr = utils.setup_graph_edges(self.DT_G, context)
            data = Data()
            data['edge_index'] = edge_index.to(config.device)
            data['edge_attr'] = edge_attr.to(config.device)
            data['y'] = utils.get_label_embedding(label, config.germeval2018_subtask_2_labels)
            data['sentence'] = context
            data.num_nodes = len(context.split())

            if(self.pre_filter is not None and not self.pre_filter(data)):
                continue 

            if(self.pre_transform is not None):
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'aux_1_{row_index}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'aux_1_{idx}.pt'))
        return data


if __name__ == '__main__':
    DT_G = utils.load_DT()

    if(config.experiment == 'WiC'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    
        WiC_train_data_frame = utils.get_WiC_data_frame(config.WiC_train_data_path, config.WiC_train_gold_path)
        train_LM_dataset = WiCLMDataset(WiC_train_data_frame, tokenizer)
        train_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        
        WiC_dev_data_frame = utils.get_WiC_data_frame(config.WiC_dev_data_path, config.WiC_dev_gold_path)
        dev_LM_dataset = WiCLMDataset(WiC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        
    elif(config.experiment == 'RTE'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
        
        RTE_train_data_frame = utils.get_RTE_data_frame(config.RTE_train_data_path)
        train_LM_dataset = RTELMDataset(RTE_train_data_frame, tokenizer)
        train_dataset_aux_1 = RTEAuxDataset(root='../data/glue_data/RTE/train/', data_frame=RTE_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = RTEAuxDataset(root='../data/glue_data/RTE/train/', data_frame=RTE_train_data_frame, DT_G=DT_G, is_sentence_2=True)

        RTE_dev_data_frame = utils.get_RTE_data_frame(config.RTE_dev_data_path)
        dev_LM_dataset = RTELMDataset(RTE_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        
    elif(config.experiment == 'STS_B'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    
        STS_B_train_data_frame = utils.get_STS_B_data_frame(config.STS_B_train_data_path)
        train_LM_dataset = STS_BLMDataset(STS_B_train_data_frame, tokenizer)
        train_dataset_aux_1 = STS_BAuxDataset('../data/glue_data/STS-B/train/', data_frame=STS_B_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = STS_BAuxDataset('../data/glue_data/STS-B/train/', data_frame=STS_B_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        
        STS_B_dev_data_frame = utils.get_STS_B_data_frame(config.STS_B_dev_data_path)
        dev_LM_dataset = STS_BLMDataset(STS_B_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
    
    elif(config.experiment == 'MRPC'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    
        MRPC_train_data_frame = utils.get_MRPC_data_frame(config.MRPC_train_data_path)
        train_LM_dataset = MRPCLMDataset(MRPC_train_data_frame, tokenizer)
        train_dataset_aux_1 = MRPCAuxDataset(root='../data/glue_data/MRPC/train/', data_frame=MRPC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = MRPCAuxDataset(root='../data/glue_data/MRPC/train/', data_frame=MRPC_train_data_frame, DT_G=DT_G, is_sentence_2=True)

        MRPC_dev_data_frame = utils.get_MRPC_data_frame(config.MRPC_dev_data_path)
        dev_LM_dataset = MRPCLMDataset(MRPC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
    
    elif(config.experiment == 'SST_2'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
        
        SST_2_train_data_frame = utils.get_SST_2_data_frame(config.SST_2_train_data_path)
        train_LM_dataset = SST_2LMDataset(SST_2_train_data_frame, tokenizer)
        train_dataset_aux_1 = SST_2AuxDataset(root='../data/glue_data/SST-2/train/', data_frame=SST_2_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        SST_2_dev_data_frame = utils.get_SST_2_data_frame(config.SST_2_dev_data_path)
        dev_LM_dataset = SST_2LMDataset(SST_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = SST_2AuxDataset(root='../data/glue_data/SST-2/dev/', data_frame=SST_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None

    elif(config.experiment == 'CoLA'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)

        CoLA_train_data_frame = utils.get_CoLA_data_frame(config.CoLA_train_data_path)
        train_LM_dataset = CoLA_LMDataset(CoLA_train_data_frame, tokenizer)
        train_dataset_aux_1 = CoLAAuxDataset(root='../data/glue_data/CoLA/train/', data_frame=CoLA_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        CoLA_dev_data_frame = utils.get_CoLA_data_frame(config.CoLA_dev_data_path)
        dev_LM_dataset = CoLA_LMDataset(CoLA_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = CoLAAuxDataset(root='../data/glue_data/CoLA/dev/', data_frame=CoLA_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None

    elif(config.experiment == 'wnli_translated'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    
        WNLI_translated_train_data_frame = utils.get_WNLI_translated_data_frame(config.WNLI_translated_train_data_path)
        train_LM_dataset = WNLI_TranslatedLMDataset(WNLI_translated_train_data_frame, tokenizer)
        train_dataset_aux_1 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/train/', data_frame=WNLI_translated_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/train/', data_frame=WNLI_translated_train_data_frame, DT_G=DT_G, is_sentence_2=True)

        WNLI_translated_dev_data_frame = utils.get_WNLI_translated_data_frame(config.WNLI_translated_dev_data_path)
        dev_LM_dataset = WNLI_TranslatedLMDataset(WNLI_translated_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
    
    elif(config.experiment == 'iitp_product'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        IITP_product_reviews_train_data_frame = utils.get_IITP_product_reviews_data_frame(config.IITP_product_reviews_train_data_path)
        train_LM_dataset = IITP_Product_ReviewsLMDataset(IITP_product_reviews_train_data_frame, tokenizer)
        train_dataset_aux_1 = IITP_Product_ReviewsAuxDataset(root='../data/iitp-product-reviews/hi/train/', data_frame=IITP_product_reviews_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        IITP_product_reviews_dev_data_frame = utils.get_IITP_product_reviews_data_frame(config.IITP_product_reviews_dev_data_path)
        dev_LM_dataset = IITP_Product_ReviewsLMDataset(IITP_product_reviews_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = IITP_Product_ReviewsAuxDataset(root='../data/iitp-product-reviews/hi/test/', data_frame=IITP_product_reviews_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'midas_discourse'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        MIDAS_discourse_train_json = utils.get_MIDAS_discourse_json(config.MIDAS_discourse_train_json_path)
        train_LM_dataset = MIDAS_DiscourseLMDataset(MIDAS_discourse_train_json, tokenizer)
        train_dataset_aux_1 = MIDAS_DiscourseAuxDataset(root='../data/midas-discourse/hi/train/', json_data=MIDAS_discourse_train_json, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        MIDAS_discourse_dev_json = utils.get_MIDAS_discourse_json(config.MIDAS_discourse_dev_json_path)
        dev_LM_dataset = MIDAS_DiscourseLMDataset(MIDAS_discourse_dev_json, tokenizer)
        dev_dataset_aux_1 = MIDAS_DiscourseAuxDataset(root='../data/midas-discourse/hi/test/', json_data=MIDAS_discourse_dev_json, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None

    elif(config.experiment == 'dpil_subtask_1'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        DPIL_subtask_1_train_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_1_train_path)
        train_LM_dataset = DPIL_Subtask_1LMDataset(DPIL_subtask_1_train_data_frame, tokenizer)
        train_dataset_aux_1 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/train/', data_frame=DPIL_subtask_1_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/train/', data_frame=DPIL_subtask_1_train_data_frame, DT_G=DT_G, is_sentence_2=True)

        DPIL_subtask_1_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_1_dev_path)
        dev_LM_dataset = DPIL_Subtask_1LMDataset(DPIL_subtask_1_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_2=True)

    elif(config.experiment == 'dpil_subtask_2'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        DPIL_subtask_2_train_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_2_train_path)
        train_LM_dataset = DPIL_Subtask_2LMDataset(DPIL_subtask_2_train_data_frame, tokenizer)
        train_dataset_aux_1 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/train/', data_frame=DPIL_subtask_2_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/train/', data_frame=DPIL_subtask_2_train_data_frame, DT_G=DT_G, is_sentence_2=True)

        DPIL_subtask_2_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_2_dev_path)
        dev_LM_dataset = DPIL_Subtask_2LMDataset(DPIL_subtask_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_2=True)

    elif(config.experiment == 'KhondokerIslam_bengali'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
        
        KhondokerIslam_bengali_train_data_frame = utils.get_KhondokerIslam_bengali_data_frame(config.KhondokerIslam_bengali_train_path)
        train_LM_dataset = KhondokerIslam_BengaliLMDataset(KhondokerIslam_bengali_train_data_frame, tokenizer)
        train_dataset_aux_1 = KhondokerIslam_BengaliAuxDataset(root='../data/KhondokerIslam_Bengali_Sentiment/train/', data_frame=KhondokerIslam_bengali_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None
        
        KhondokerIslam_bengali_dev_data_frame = utils.get_KhondokerIslam_bengali_data_frame(config.KhondokerIslam_bengali_dev_path)
        dev_LM_dataset = KhondokerIslam_BengaliLMDataset(KhondokerIslam_bengali_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = KhondokerIslam_BengaliAuxDataset(root='../data/KhondokerIslam_Bengali_Sentiment/test/', data_frame=KhondokerIslam_bengali_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'rezacsedu_sentiment'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
        
        rezacsedu_sentiment_train_data_frame = utils.get_rezacsedu_sentiment_data_frame(config.rezacsedu_sentiment_train_path)
        train_LM_dataset = Rezacsedu_SentimentLMDataset(rezacsedu_sentiment_train_data_frame, tokenizer)
        train_dataset_aux_1 = Rezacsedu_SentimentAuxDataset(root='../data/rezacsedu_sentiment/train/', data_frame=rezacsedu_sentiment_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        rezacsedu_sentiment_dev_data_frame = utils.get_rezacsedu_sentiment_data_frame(config.rezacsedu_sentiment_test_path)
        dev_LM_dataset = Rezacsedu_SentimentLMDataset(rezacsedu_sentiment_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Rezacsedu_SentimentAuxDataset(root='../data/rezacsedu_sentiment/test/', data_frame=rezacsedu_sentiment_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'BEmoC'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        BEmoC_train_data_frame = utils.get_BEmoC_data_frame(config.BEmoC_train_path)
        train_LM_dataset = BEmoCLMDataset(BEmoC_train_data_frame, tokenizer)
        train_dataset_aux_1 = BEmoCLMDatasetAuxDataset(root='../data/BEmoC/train/', data_frame=BEmoC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        BEmoC_dev_data_frame = utils.get_BEmoC_data_frame(config.BEmoC_dev_path)
        dev_LM_dataset = BEmoCLMDataset(BEmoC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = BEmoCLMDatasetAuxDataset(root='../data/BEmoC/test/', data_frame=BEmoC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None

    elif(config.experiment == 'seid_amharic_sentiment'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        Seid_amharic_train_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_train_path)
        train_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_train_data_frame, tokenizer)
        train_dataset_aux_1 = Seid_Amharic_SentimentAuxDataset(root='../data/seid_amharic_sentiment/train/', data_frame=Seid_amharic_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'seid_amharic_cleaned_sentiment'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        Seid_amharic_train_data_frame = utils.get_Seid_amharic_sentiment_cleaned_data_frame(config.Seid_Amharic_Sentiment_train_path)
        train_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_train_data_frame, tokenizer)
        train_dataset_aux_1 = Seid_Amharic_Cleaned_SentimentAuxDataset(root='../data/seid_amharic_sentiment/train/', data_frame=Seid_amharic_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None

        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_cleaned_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_Cleaned_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'germeval2018_subtask_1'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        germeval2018_subtask_train_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_train_path)
        train_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_train_data_frame, tokenizer)
        train_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/train/', data_frame=germeval2018_subtask_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None
        
        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
    
    elif(config.experiment == 'germeval2018_subtask_2'):
        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)

        germeval2018_subtask_train_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_train_path)
        train_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_train_data_frame, tokenizer)
        train_dataset_aux_1 = Germeval2018_Subtask_2AuxDataset(root='../data/germeval2018/train/', data_frame=germeval2018_subtask_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = None
        
        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_2AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = None
        
    if(train_dataset_aux_2 is not None):
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)
    else:
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)
    batch = next(iter(train_loader))
    if(len(batch) == 3):
        [[input_ids, token_type_ids, attention_mask], batch_aux_1, batch_aux_2] = batch
    else:
        [[input_ids, token_type_ids, attention_mask], batch_aux_1] = batch
        batch_aux_2 = None
    print(f'input_ids.shape: {input_ids.shape}')
    print(f'token_type_ids.shape: {token_type_ids.shape}')
    print(f'attention_mask.shape: { attention_mask.shape}')
    print(f'batch_aux_1: {batch_aux_1}')
    if(batch_aux_2 is not None):
        print(f'batch_aux_2: {batch_aux_2}')

    if(dev_dataset_aux_2 is not None):
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)
    else:
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)
    batch = next(iter(dev_loader))
    if(len(batch) == 3):
        [[input_ids, token_type_ids, attention_mask], batch_aux_1, batch_aux_2] = batch
    else:
        [[input_ids, token_type_ids, attention_mask], batch_aux_1] = batch
        batch_aux_2 = None
    print(f'input_ids.shape: {input_ids.shape}')
    print(f'token_type_ids.shape: {token_type_ids.shape}')
    print(f'attention_mask.shape: { attention_mask.shape}')
    print(f'batch_aux_1: {batch_aux_1}')
    if(batch_aux_2 is not None):
        print(f'batch_aux_2: {batch_aux_2}')
