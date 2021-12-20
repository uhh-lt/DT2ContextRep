import os 
import random
from tqdm import tqdm

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr
import numpy as np
import torch 
import torch.nn as nn
from torch_geometric.data import DataLoader as GraphDataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import config
import utils
from data_loader import WrapperDataset, WiCLMDataset, WiCAuxDataset, RTELMDataset, RTEAuxDataset, STS_BLMDataset, STS_BAuxDataset, MRPCLMDataset, MRPCAuxDataset, SST_2LMDataset, SST_2AuxDataset, WNLI_TranslatedLMDataset, WNLI_TranslatedAuxDataset, IITP_Product_ReviewsLMDataset, IITP_Product_ReviewsAuxDataset, MIDAS_DiscourseLMDataset, MIDAS_DiscourseAuxDataset, DPIL_Subtask_1LMDataset, DPIL_Subtask_1AuxDataset, DPIL_Subtask_2LMDataset, DPIL_Subtask_2AuxDataset, CoLA_LMDataset, CoLAAuxDataset, KhondokerIslam_BengaliLMDataset, KhondokerIslam_BengaliAuxDataset, Rezacsedu_SentimentLMDataset, Rezacsedu_SentimentAuxDataset, BEmoCLMDataset, BEmoCLMDatasetAuxDataset, Seid_Amharic_SentimentLMDataset, Seid_Amharic_SentimentAuxDataset, Seid_Amharic_Cleaned_SentimentAuxDataset, Germeval2018LMDataset, Germeval2018_Subtask_1AuxDataset, Germeval2018_Subtask_2AuxDataset
from model import WordLevelNet, SentenceLevelNet

seed_dict = {1: 42, 2: 98, 3: 3, 4: 9, 5: 7}
seed = seed_dict[config.exp_no]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(model, train_loader, loss_fn, optimizer, scheduler, dev_loader, epochs=config.epochs):
    writer = SummaryWriter(config.model_folder)
    if(config.scoring == 'loss'):
        best_val_score = np.inf
    else:
        best_val_score = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_actual_list = list()
        y_pred_list = list()

        print(f'Epoch: {epoch}')
        for batch in tqdm(train_loader):
            if(len(batch) == 3):
                [input_ids, token_type_ids, attention_mask], batch_aux_1, batch_aux_2 = batch
            else:
                [input_ids, token_type_ids, attention_mask], batch_aux_1 = batch
                batch_aux_2 = None
            optimizer.zero_grad()
            if(batch_aux_2 is not None):
                assert(torch.equal(batch_aux_1.y, batch_aux_2.y))
            if(config.is_classification_task):
                y_actual = torch.argmax(batch_aux_1.y, dim=1)
                y_actual_list += list(y_actual.cpu().data.numpy())
                if(config.debug == True):
                    latent, y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
                elif(config.debug == False):
                    y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
                # assert(torch.isnan(y_out).any() == False)
                output_probs = y_out
                y_pred = torch.argmax(output_probs, dim=1)
                y_pred_list += list(y_pred.cpu().data.numpy())
            else:
                y_actual = batch_aux_1.y
                y_actual_list += list(y_actual.cpu().data.numpy().flatten())
                if(config.debug == True):
                    latent, y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
                elif(config.debug == False):
                    y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
                y_pred_list += list(y_out.cpu().data.numpy().flatten())

            loss = loss_fn(y_out, y_actual)
            loss.backward()
            if(config.lm_model_name.startswith('bert') or config.lm_model_name.startswith('albert')):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
        
        if(config.is_classification_task):
            accuracy = accuracy_score(y_actual_list, y_pred_list) * 100
            mcc = matthews_corrcoef(y_actual_list, y_pred_list) * 100
            macro_f1 = f1_score(y_actual_list, y_pred_list, average='macro') * 100
        else:
            pearson, _ = pearsonr(y_actual_list, y_pred_list)
            pearson = pearson * 100

        writer.add_scalar('Loss/train', total_loss/len(train_loader), epoch)
        if(config.is_classification_task):
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Matthews/train', mcc, epoch)
            writer.add_scalar('Macro_F1/train', macro_f1, epoch)
        else:
            writer.add_scalar('Pearson/train', pearson, epoch)

        print(f'Training scores')
        if(config.is_classification_task):
            print(f'Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}, Matthews: {mcc}, Macro_F1: {macro_f1}')
        else:
            print(f'Loss: {total_loss/len(train_loader)}, Pearson: {pearson}')

        val_scores = dict()
        if(config.is_classification_task):
            val_scores['loss'], val_scores['accuracy'], val_scores['matthews'], val_scores['macro_f1'], predictions, actuals = test(model=model, test_loader=dev_loader, loss_fn=loss_fn, writer=writer, epoch=epoch)
        else:
            val_scores['loss'], val_scores['pearson'] = test(model=model, test_loader=dev_loader, loss_fn=loss_fn, writer=writer, epoch=epoch)

        if(config.scoring == 'loss'):
            if(val_scores[config.scoring] <= best_val_score):
                best_model = model
                best_val_loss = val_scores['loss']
                if(config.is_classification_task):
                    best_val_accuracy = val_scores['accuracy']
                    best_val_mcc = val_scores['matthews']
                    best_val_macro_f1 = val_scores['macro_f1']
                else:
                    best_val_pearson = val_scores['pearson']
                best_val_score = best_val_loss
                torch.save(model.state_dict(), config.model_name)
        else:
            if(val_scores[config.scoring] >= best_val_score):
                best_model = model
                best_val_loss = val_scores['loss']
                if(config.is_classification_task):
                    best_val_accuracy = val_scores['accuracy']
                    best_val_mcc = val_scores['matthews']
                    best_val_macro_f1 = val_scores['macro_f1']
                else:
                    best_val_pearson = val_scores['pearson']
                if(config.scoring == 'accuracy'):
                    best_val_score = best_val_accuracy
                elif(config.scoring == 'matthews'):
                    best_val_score = best_val_mcc
                elif(config.scoring == 'pearson'):
                    best_val_score = best_val_pearson
                elif(config.scoring == 'macro_f1'):
                    best_val_score = best_val_macro_f1
                torch.save(model.state_dict(), config.model_name)
    
    writer.close()
    if(config.is_classification_task):
        print(f'Scoring: {config.scoring}, Validation Loss: {best_val_loss}, Validation Accuracy: {best_val_accuracy}, Validation Matthews: {best_val_mcc}, Validation Macro_F1: {best_val_macro_f1}')
    else:
        print(f'Scoring: {config.scoring}, Validation Loss: {best_val_loss}, Validation Pearson: {best_val_pearson}')
    return best_model

def test(model, test_loader, loss_fn, writer=None, epoch=None):
    model.eval()
    total_loss = 0
    y_actual_list = list()
    y_pred_list = list()

    for batch in tqdm(test_loader):
        if(len(batch) == 3):
            [input_ids, token_type_ids, attention_mask], batch_aux_1, batch_aux_2 = batch
        else:
            [input_ids, token_type_ids, attention_mask], batch_aux_1 = batch
            batch_aux_2 = None
        if(batch_aux_2 is not None):
            assert(torch.equal(batch_aux_1.y, batch_aux_2.y))
        if(config.is_classification_task):
            y_actual = torch.argmax(batch_aux_1.y, dim=1)
            y_actual_list += list(y_actual.cpu().data.numpy())
            if(config.debug == True):
                latent, y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
            elif(config.debug == False):
                y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
            output_probs = y_out
            y_pred = torch.argmax(output_probs, dim=1)
            y_pred_list += list(y_pred.cpu().data.numpy())
        else:
            y_actual = batch_aux_1.y
            y_actual_list += list(y_actual.cpu().data.numpy().flatten())
            if(config.debug == True):
                latent, y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
            elif(config.debug == False):
                y_out = model(input_ids, token_type_ids, attention_mask, batch_aux_1, batch_aux_2)
            y_pred_list += list(y_out.cpu().data.numpy().flatten())
            
        loss = loss_fn(y_out, y_actual)
        total_loss += loss.item()
    
    if(config.is_classification_task):
        accuracy = accuracy_score(y_actual_list, y_pred_list) * 100
        mcc = matthews_corrcoef(y_actual_list, y_pred_list) * 100
        macro_f1 = f1_score(y_actual_list, y_pred_list, average='macro') * 100
    else:
        pearson, _ = pearsonr(y_actual_list, y_pred_list)
        pearson = pearson * 100

    if(writer and epoch):
        writer.add_scalar('Loss/test', total_loss/len(test_loader), epoch)
        if(config.is_classification_task):
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            writer.add_scalar('Matthews/test', mcc, epoch)
            writer.add_scalar('Macro_F1/test', macro_f1, epoch)
        else:
            writer.add_scalar('Pearson/test', pearson, epoch)
    
    print(f'Testing scores')
    if(config.is_classification_task):
        print(f'Loss: {total_loss/len(test_loader)}, Accuracy: {accuracy}, Matthews: {mcc}, Macro_F1: {macro_f1}')
        return total_loss/len(test_loader), accuracy, mcc, macro_f1, y_pred_list, y_actual_list
    else:
        print(f'Loss: {total_loss/len(test_loader)}, Pearson: {pearson}')
        return total_loss/len(test_loader), pearson

def WiC_main():
    assert(config.experiment == 'WiC')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = WordLevelNet(tokenizer=tokenizer, num_output_classes=len(config.WiC_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        WiC_train_data_frame = utils.get_WiC_data_frame(config.WiC_train_data_path, config.WiC_train_gold_path)
        train_LM_dataset = WiCLMDataset(WiC_train_data_frame, tokenizer)
        train_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/train/', data_frame=WiC_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)

        WiC_dev_data_frame = utils.get_WiC_data_frame(config.WiC_dev_data_path, config.WiC_dev_gold_path)
        dev_LM_dataset = WiCLMDataset(WiC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        WiC_dev_data_frame = utils.get_WiC_data_frame(config.WiC_dev_data_path, config.WiC_dev_gold_path)
        dev_LM_dataset = WiCLMDataset(WiC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WiCAuxDataset(root='../data/WiC/dev/', data_frame=WiC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)
        
        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def RTE_main():
    assert(config.experiment == 'RTE')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.RTE_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        RTE_train_data_frame = utils.get_RTE_data_frame(config.RTE_train_data_path)
        train_LM_dataset = RTELMDataset(RTE_train_data_frame, tokenizer)
        train_dataset_aux_1 = RTEAuxDataset(root='../data/glue_data/RTE/train/', data_frame=RTE_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = RTEAuxDataset(root='../data/glue_data/RTE/train/', data_frame=RTE_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)
        
        RTE_dev_data_frame = utils.get_RTE_data_frame(config.RTE_dev_data_path)
        dev_LM_dataset = RTELMDataset(RTE_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        RTE_dev_data_frame = utils.get_RTE_data_frame(config.RTE_dev_data_path)
        dev_LM_dataset = RTELMDataset(RTE_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = RTEAuxDataset(root='../data/glue_data/RTE/dev/', data_frame=RTE_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)
        
        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def STS_B_main():
    assert(config.experiment == 'STS_B')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=1)
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        STS_B_train_data_frame = utils.get_STS_B_data_frame(config.STS_B_train_data_path)
        train_LM_dataset = STS_BLMDataset(STS_B_train_data_frame, tokenizer)
        train_dataset_aux_1 = STS_BAuxDataset('../data/glue_data/STS-B/train/', data_frame=STS_B_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = STS_BAuxDataset('../data/glue_data/STS-B/train/', data_frame=STS_B_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)
        
        STS_B_dev_data_frame = utils.get_STS_B_data_frame(config.STS_B_dev_data_path)
        dev_LM_dataset = STS_BLMDataset(STS_B_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.MSELoss()

        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        STS_B_dev_data_frame = utils.get_STS_B_data_frame(config.STS_B_dev_data_path)
        dev_LM_dataset = STS_BLMDataset(STS_B_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = STS_BAuxDataset('../data/glue_data/STS-B/dev/', data_frame=STS_B_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        loss_fn = nn.MSELoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def MRPC_main():
    assert(config.experiment == 'MRPC')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.MRPC_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        MRPC_train_data_frame = utils.get_MRPC_data_frame(config.MRPC_train_data_path)
        train_LM_dataset = MRPCLMDataset(MRPC_train_data_frame, tokenizer)
        train_dataset_aux_1 = MRPCAuxDataset(root='../data/glue_data/MRPC/train/', data_frame=MRPC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = MRPCAuxDataset(root='../data/glue_data/MRPC/train/', data_frame=MRPC_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)

        MRPC_dev_data_frame = utils.get_MRPC_data_frame(config.MRPC_dev_data_path)
        dev_LM_dataset = MRPCLMDataset(MRPC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        MRPC_dev_data_frame = utils.get_MRPC_data_frame(config.MRPC_dev_data_path)
        dev_LM_dataset = MRPCLMDataset(MRPC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = MRPCAuxDataset(root='../data/glue_data/MRPC/dev/', data_frame=MRPC_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def SST_2_main():
    assert(config.experiment == 'SST_2')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=True)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.SST_2_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        SST_2_train_data_frame = utils.get_SST_2_data_frame(config.SST_2_train_data_path)
        train_LM_dataset = SST_2LMDataset(SST_2_train_data_frame, tokenizer)
        train_dataset_aux_1 = SST_2AuxDataset(root='../data/glue_data/SST-2/train/', data_frame=SST_2_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        SST_2_dev_data_frame = utils.get_SST_2_data_frame(config.SST_2_dev_data_path)
        dev_LM_dataset = SST_2LMDataset(SST_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = SST_2AuxDataset(root='../data/glue_data/SST-2/dev/', data_frame=SST_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        SST_2_dev_data_frame = utils.get_SST_2_data_frame(config.SST_2_dev_data_path)
        dev_LM_dataset = SST_2LMDataset(SST_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = SST_2AuxDataset(root='../data/glue_data/SST-2/dev/', data_frame=SST_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def CoLA_main():
    assert(config.experiment == 'CoLA')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.CoLA_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        CoLA_train_data_frame = utils.get_CoLA_data_frame(config.CoLA_train_data_path)
        train_LM_dataset = CoLA_LMDataset(CoLA_train_data_frame, tokenizer)
        train_dataset_aux_1 = CoLAAuxDataset(root='../data/glue_data/CoLA/train/', data_frame=CoLA_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        CoLA_dev_data_frame = utils.get_CoLA_data_frame(config.CoLA_dev_data_path)
        dev_LM_dataset = CoLA_LMDataset(CoLA_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = CoLAAuxDataset(root='../data/glue_data/CoLA/dev/', data_frame=CoLA_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        CoLA_dev_data_frame = utils.get_CoLA_data_frame(config.CoLA_dev_data_path)
        dev_LM_dataset = CoLA_LMDataset(CoLA_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = CoLAAuxDataset(root='../data/glue_data/CoLA/dev/', data_frame=CoLA_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def WNLI_translated_main():
    assert(config.experiment == 'wnli_translated')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.SST_2_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        WNLI_translated_train_data_frame = utils.get_WNLI_translated_data_frame(config.WNLI_translated_train_data_path)
        train_LM_dataset = WNLI_TranslatedLMDataset(WNLI_translated_train_data_frame, tokenizer)
        train_dataset_aux_1 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/train/', data_frame=WNLI_translated_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/train/', data_frame=WNLI_translated_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)

        WNLI_translated_dev_data_frame = utils.get_WNLI_translated_data_frame(config.WNLI_translated_dev_data_path)
        dev_LM_dataset = WNLI_TranslatedLMDataset(WNLI_translated_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        WNLI_translated_dev_data_frame = utils.get_WNLI_translated_data_frame(config.WNLI_translated_dev_data_path)
        dev_LM_dataset = WNLI_TranslatedLMDataset(WNLI_translated_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = WNLI_TranslatedAuxDataset(root='../data/wnli-translated/hi/dev/', data_frame=WNLI_translated_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def IITP_product_reviews_main():
    assert(config.experiment == 'iitp_product')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.IITP_product_reviews_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        IITP_product_reviews_train_data_frame = utils.get_IITP_product_reviews_data_frame(config.IITP_product_reviews_train_data_path)
        train_LM_dataset = IITP_Product_ReviewsLMDataset(IITP_product_reviews_train_data_frame, tokenizer)
        train_dataset_aux_1 = IITP_Product_ReviewsAuxDataset(root='../data/iitp-product-reviews/hi/train/', data_frame=IITP_product_reviews_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        IITP_product_reviews_dev_data_frame = utils.get_IITP_product_reviews_data_frame(config.IITP_product_reviews_dev_data_path)
        dev_LM_dataset = IITP_Product_ReviewsLMDataset(IITP_product_reviews_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = IITP_Product_ReviewsAuxDataset(root='../data/iitp-product-reviews/hi/test/', data_frame=IITP_product_reviews_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        IITP_product_reviews_dev_data_frame = utils.get_IITP_product_reviews_data_frame(config.IITP_product_reviews_dev_data_path)
        dev_LM_dataset = IITP_Product_ReviewsLMDataset(IITP_product_reviews_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = IITP_Product_ReviewsAuxDataset(root='../data/iitp-product-reviews/hi/valid/', data_frame=IITP_product_reviews_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def MIDAS_discourse_main():
    assert(config.experiment == 'midas_discourse')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.MIDAS_discourse_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        MIDAS_discourse_train_json = utils.get_MIDAS_discourse_json(config.MIDAS_discourse_train_json_path)
        train_LM_dataset = MIDAS_DiscourseLMDataset(MIDAS_discourse_train_json, tokenizer)
        train_dataset_aux_1 = MIDAS_DiscourseAuxDataset(root='../data/midas-discourse/hi/train/', json_data=MIDAS_discourse_train_json, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        MIDAS_discourse_dev_json = utils.get_MIDAS_discourse_json(config.MIDAS_discourse_dev_json_path)
        dev_LM_dataset = MIDAS_DiscourseLMDataset(MIDAS_discourse_dev_json, tokenizer)
        dev_dataset_aux_1 = MIDAS_DiscourseAuxDataset(root='../data/midas-discourse/hi/test/', json_data=MIDAS_discourse_dev_json, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        MIDAS_discourse_dev_json = utils.get_MIDAS_discourse_json(config.MIDAS_discourse_dev_json_path)
        dev_LM_dataset = MIDAS_DiscourseLMDataset(MIDAS_discourse_dev_json, tokenizer)
        dev_dataset_aux_1 = MIDAS_DiscourseAuxDataset(root='../data/midas-discourse/hi/dev/', json_data=MIDAS_discourse_dev_json, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def DPIL_subtask_1_main():
    assert(config.experiment == 'dpil_subtask_1')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.DPIL_subtask_1_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        DPIL_subtask_1_train_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_1_train_path)
        train_LM_dataset = DPIL_Subtask_1LMDataset(DPIL_subtask_1_train_data_frame, tokenizer)
        train_dataset_aux_1 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/train/', data_frame=DPIL_subtask_1_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/train/', data_frame=DPIL_subtask_1_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)

        DPIL_subtask_1_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_1_dev_path)
        dev_LM_dataset = DPIL_Subtask_1LMDataset(DPIL_subtask_1_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        DPIL_subtask_1_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_1_dev_path)
        dev_LM_dataset = DPIL_Subtask_1LMDataset(DPIL_subtask_1_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_1AuxDataset(root='../data/DPIL_csv/subtask_1/test/', data_frame=DPIL_subtask_1_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def DPIL_subtask_2_main():
    assert(config.experiment == 'dpil_subtask_2')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.DPIL_subtask_2_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        DPIL_subtask_2_train_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_2_train_path)
        train_LM_dataset = DPIL_Subtask_2LMDataset(DPIL_subtask_2_train_data_frame, tokenizer)
        train_dataset_aux_1 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/train/', data_frame=DPIL_subtask_2_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_dataset_aux_2 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/train/', data_frame=DPIL_subtask_2_train_data_frame, DT_G=DT_G, is_sentence_2=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1, train_dataset_aux_2), batch_size=config.batch_size)

        DPIL_subtask_2_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_2_dev_path)
        dev_LM_dataset = DPIL_Subtask_2LMDataset(DPIL_subtask_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        print(config.model_name)
        DPIL_subtask_2_dev_data_frame = utils.get_DPIL_data_frame(config.DPIL_subtask_2_dev_path)
        dev_LM_dataset = DPIL_Subtask_2LMDataset(DPIL_subtask_2_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_dataset_aux_2 = DPIL_Subtask_2AuxDataset(root='../data/DPIL_csv/subtask_2/test/', data_frame=DPIL_subtask_2_dev_data_frame, DT_G=DT_G, is_sentence_2=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1, dev_dataset_aux_2), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def KhondokerIslam_Bengali_main():
    assert(config.experiment == 'KhondokerIslam_bengali')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.KhondokerIslam_bengali_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        KhondokerIslam_bengali_train_data_frame = utils.get_KhondokerIslam_bengali_data_frame(config.KhondokerIslam_bengali_train_path)
        train_LM_dataset = KhondokerIslam_BengaliLMDataset(KhondokerIslam_bengali_train_data_frame, tokenizer)
        train_dataset_aux_1 = KhondokerIslam_BengaliAuxDataset(root='../data/KhondokerIslam_Bengali_Sentiment/train/', data_frame=KhondokerIslam_bengali_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        KhondokerIslam_bengali_dev_data_frame = utils.get_KhondokerIslam_bengali_data_frame(config.KhondokerIslam_bengali_dev_path)
        dev_LM_dataset = KhondokerIslam_BengaliLMDataset(KhondokerIslam_bengali_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = KhondokerIslam_BengaliAuxDataset(root='../data/KhondokerIslam_Bengali_Sentiment/test/', data_frame=KhondokerIslam_bengali_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        KhondokerIslam_bengali_dev_data_frame = utils.get_KhondokerIslam_bengali_data_frame(config.KhondokerIslam_bengali_test_path)
        dev_LM_dataset = KhondokerIslam_BengaliLMDataset(KhondokerIslam_bengali_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = KhondokerIslam_BengaliAuxDataset(root='../data/KhondokerIslam_Bengali_Sentiment/test/', data_frame=KhondokerIslam_bengali_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def Rezacsedu_Sentiment_main():
    assert(config.experiment == 'rezacsedu_sentiment')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.rezacsedu_sentiment_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        rezacsedu_sentiment_train_data_frame = utils.get_rezacsedu_sentiment_data_frame(config.rezacsedu_sentiment_train_path)
        train_LM_dataset = Rezacsedu_SentimentLMDataset(rezacsedu_sentiment_train_data_frame, tokenizer)
        train_dataset_aux_1 = Rezacsedu_SentimentAuxDataset(root='../data/rezacsedu_sentiment/train/', data_frame=rezacsedu_sentiment_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)
        
        rezacsedu_sentiment_dev_data_frame = utils.get_rezacsedu_sentiment_data_frame(config.rezacsedu_sentiment_test_path)
        dev_LM_dataset = Rezacsedu_SentimentLMDataset(rezacsedu_sentiment_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Rezacsedu_SentimentAuxDataset(root='../data/rezacsedu_sentiment/test/', data_frame=rezacsedu_sentiment_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        rezacsedu_sentiment_dev_data_frame = utils.get_rezacsedu_sentiment_data_frame(config.rezacsedu_sentiment_test_path)
        dev_LM_dataset = Rezacsedu_SentimentLMDataset(rezacsedu_sentiment_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Rezacsedu_SentimentAuxDataset(root='../data/rezacsedu_sentiment/test/', data_frame=rezacsedu_sentiment_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def BEmoC_main():
    assert(config.experiment == 'BEmoC')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.BEmoC_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        BEmoC_train_data_frame = utils.get_BEmoC_data_frame(config.BEmoC_train_path)
        train_LM_dataset = BEmoCLMDataset(BEmoC_train_data_frame, tokenizer)
        train_dataset_aux_1 = BEmoCLMDatasetAuxDataset(root='../data/BEmoC/train/', data_frame=BEmoC_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)
        
        BEmoC_dev_data_frame = utils.get_BEmoC_data_frame(config.BEmoC_dev_path)
        dev_LM_dataset = BEmoCLMDataset(BEmoC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = BEmoCLMDatasetAuxDataset(root='../data/BEmoC/test/', data_frame=BEmoC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        BEmoC_dev_data_frame = utils.get_BEmoC_data_frame(config.BEmoC_dev_path)
        dev_LM_dataset = BEmoCLMDataset(BEmoC_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = BEmoCLMDatasetAuxDataset(root='../data/BEmoC/test/', data_frame=BEmoC_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def Seid_Amharic_Sentiment_main():
    assert(config.experiment == 'seid_amharic_sentiment')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.Seid_Amharic_Sentiment_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        Seid_amharic_train_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_train_path)
        train_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_train_data_frame, tokenizer)
        train_dataset_aux_1 = Seid_Amharic_SentimentAuxDataset(root='../data/seid_amharic_sentiment/train/', data_frame=Seid_amharic_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def Seid_Amharic_Cleaned_Sentiment_main():
    assert(config.experiment == 'seid_amharic_cleaned_sentiment')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.Seid_Amharic_Sentiment_cleaned_labels))
    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        Seid_amharic_train_data_frame = utils.get_Seid_amharic_sentiment_cleaned_data_frame(config.Seid_Amharic_Sentiment_train_path)
        train_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_train_data_frame, tokenizer)
        train_dataset_aux_1 = Seid_Amharic_Cleaned_SentimentAuxDataset(root='../data/seid_amharic_sentiment/train/', data_frame=Seid_amharic_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_cleaned_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_Cleaned_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        Seid_amharic_dev_data_frame = utils.get_Seid_amharic_sentiment_data_frame(config.Seid_Amharic_Sentiment_dev_path)
        dev_LM_dataset = Seid_Amharic_SentimentLMDataset(Seid_amharic_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Seid_Amharic_Cleaned_SentimentAuxDataset(root='../data/seid_amharic_sentiment/test/', data_frame=Seid_amharic_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(config.model_name))

        print(f'Testing')
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def Germeval2018_Subtask_1_main():
    assert(config.experiment == 'germeval2018_subtask_1')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.germeval2018_subtask_1_labels))

    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        germeval2018_subtask_train_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_train_path)
        train_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_train_data_frame, tokenizer)
        train_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/train/', data_frame=germeval2018_subtask_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def Germeval2018_Subtask_2_main():
    assert(config.experiment == 'germeval2018_subtask_2')
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, do_lower_case=False)
    DT_G = utils.load_DT()
    model = SentenceLevelNet(tokenizer=tokenizer, num_output_classes=len(config.germeval2018_subtask_2_labels))

    model.to(config.device)

    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if(config.train):
        germeval2018_subtask_train_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_train_path)
        train_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_train_data_frame, tokenizer)
        train_dataset_aux_1 = Germeval2018_Subtask_2AuxDataset(root='../data/germeval2018/train/', data_frame=germeval2018_subtask_train_data_frame, DT_G=DT_G, is_sentence_1=True)
        train_loader = GraphDataLoader(WrapperDataset(train_LM_dataset, train_dataset_aux_1), batch_size=config.batch_size)

        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_2AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],},]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
        t_total = (len(train_loader.dataset) // config.batch_size) * float(config.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_ratio*t_total, num_training_steps=t_total)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f'Training: {config.experiment}')
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, dev_loader=dev_loader)
    
    else:
        germeval2018_subtask_dev_data_frame = utils.get_germeval2018_data_frame(config.germeval2018_dev_path)
        dev_LM_dataset = Germeval2018LMDataset(germeval2018_subtask_dev_data_frame, tokenizer)
        dev_dataset_aux_1 = Germeval2018_Subtask_1AuxDataset(root='../data/germeval2018/test/', data_frame=germeval2018_subtask_dev_data_frame, DT_G=DT_G, is_sentence_1=True)
        dev_loader = GraphDataLoader(WrapperDataset(dev_LM_dataset, dev_dataset_aux_1), batch_size=config.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        return test(model=model, test_loader=dev_loader, loss_fn=loss_fn)

def choose_main():
    if(config.experiment == 'WiC'):
        return WiC_main()
    elif(config.experiment == 'RTE'):
        # removed " from sentence 1 of line 2166 (index: 2164)
        return RTE_main()
    elif(config.experiment == 'STS_B'):
        return STS_B_main()
    elif(config.experiment == 'MRPC'):
        return MRPC_main()
    elif(config.experiment == 'SST_2'):
        return SST_2_main()
    elif(config.experiment == 'CoLA'):
        return CoLA_main()
    elif(config.experiment == 'wnli_translated'):
        return WNLI_translated_main()
    elif(config.experiment == 'iitp_product'):
        return IITP_product_reviews_main()
    elif(config.experiment == 'midas_discourse'):
        return MIDAS_discourse_main()
    elif(config.experiment == 'dpil_subtask_1'):
        return DPIL_subtask_1_main()
    elif(config.experiment == 'dpil_subtask_2'):
        return DPIL_subtask_2_main()
    elif(config.experiment == 'KhondokerIslam_bengali'):
        return KhondokerIslam_Bengali_main()
    elif(config.experiment == 'rezacsedu_sentiment'):
        return Rezacsedu_Sentiment_main()
    elif(config.experiment == 'BEmoC'):
        return BEmoC_main()
    elif(config.experiment == 'seid_amharic_sentiment'):
        return Seid_Amharic_Sentiment_main()
    elif(config.experiment == 'seid_amharic_cleaned_sentiment'):
        return Seid_Amharic_Cleaned_Sentiment_main()
    elif(config.experiment == 'germeval2018_subtask_1'):
        return Germeval2018_Subtask_1_main()
    elif(config.experiment == 'germeval2018_subtask_2'):
        return Germeval2018_Subtask_2_main()

if __name__ == '__main__':
    choose_main()
