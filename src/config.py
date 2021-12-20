import os
import torch

is_en = False
is_hi = False
is_bn = False
is_am = True
is_de = False
assert(sum([is_en, is_hi, is_bn, is_am, is_de]) == 1)
assert(sum([is_hi, is_bn, is_am, is_de]) == 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
train = False
exp_no = 1
experiment = 'seid_amharic_cleaned_sentiment'
lm_model_name = 'xlm-roberta-base'
debug = True

learning_rate = 2e-5
warmup_ratio = 0.1
dropout = 0.1

path_lambda = 1
assert(path_lambda == 1)
use_gcn = False
use_le = False

if(experiment == 'WiC'):
    batch_size = 2
    scoring = 'accuracy'
    is_sentence_pair_task = True
    is_classification_task = True
elif(experiment == 'RTE'):
    batch_size = 1
    scoring = 'accuracy'
    is_classification_task = True
    is_sentence_pair_task = True
elif(experiment == 'STS_B'):
    batch_size = 4
    scoring = 'pearson'
    is_sentence_pair_task = True
    is_classification_task = False
elif(experiment == 'MRPC'):
    batch_size = 4
    scoring = 'accuracy'
    is_sentence_pair_task = True
    is_classification_task = True
elif(experiment == 'SST_2'):
    batch_size = 1
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'CoLA'):
    batch_size = 1
    scoring = 'matthews'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'wnli_translated'):
    batch_size = 4
    scoring = 'accuracy'
    is_sentence_pair_task = True
    is_classification_task = True
elif(experiment == 'iitp_product'):
    batch_size = 4
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'midas_discourse'):
    batch_size = 8
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'dpil_subtask_1'):
    batch_size = 16
    scoring = 'accuracy'
    is_sentence_pair_task = True
    is_classification_task = True
elif(experiment == 'dpil_subtask_2'):
    batch_size = 16
    scoring = 'accuracy'
    is_sentence_pair_task = True
    is_classification_task = True
elif(experiment == 'KhondokerIslam_bengali'):
    batch_size = 2
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'rezacsedu_sentiment'):
    batch_size = 4
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'BEmoC'):
    batch_size = 4
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'seid_amharic_sentiment'):
    batch_size = 2
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'seid_amharic_cleaned_sentiment'):
    batch_size = 2
    scoring = 'accuracy'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'germeval2018_subtask_1'):
    batch_size = 4
    scoring = 'macro_f1'
    is_sentence_pair_task = False
    is_classification_task = True
elif(experiment == 'germeval2018_subtask_2'):
    batch_size = 4
    scoring = 'macro_f1'
    is_sentence_pair_task = False
    is_classification_task = True

if(debug == True):
    batch_size = 1

if(is_en):
    DT_edges_path = '../data/DT/en_news120M_stanford_lemma/LMI_p1000_l200'
    DT_word_count_path = '../data/DT/en_news120M_stanford_lemma/word_count'
    DT_token_pos_delimiter = '#'

    prune_DT_frequency_threshold = 100
    prune_DT_filename = f'prune_{prune_DT_frequency_threshold}_LMI_p1000_l200'
    prune_DT_edges_path = f'../data/DT/en_news120M_stanford_lemma/{prune_DT_filename}'
    prune_DT_update_POS_filename = f'{prune_DT_filename}_update_POS'
    prune_DT_update_POS_edges_path = f'../data/DT/en_news120M_stanford_lemma/{prune_DT_update_POS_filename}'
    prune_DT_filename_orig = prune_DT_filename
    prune_DT_edges_path_orig = prune_DT_edges_path
    update_POS = True

    if(update_POS):
        prune_DT_filename = prune_DT_update_POS_filename
        prune_DT_edges_path = prune_DT_update_POS_edges_path

else:
    if(is_hi):
        DT_edges_path = '../data/DT/hi/hindi_bigram__PruneFeaturesPerWord_1000__FreqSigLMI_s_0_t_0__PruneGraph_p_1000__AggrPerFt__SimCounts1WithFeatures__SimSortlimit_200'
        DT_word_count_path = '../data/DT/hi/hindi_bigram__PruneFeaturesPerWord_1000__WordCount'

        prune_DT_frequency_threshold = 100
        prune_DT_filename = f'prune_{prune_DT_frequency_threshold}_hindi_bigram__PruneFeaturesPerWord_1000__FreqSigLMI_s_0_t_0__PruneGraph_p_1000__AggrPerFt__SimCounts1WithFeatures__SimSortlimit_200'
        prune_DT_edges_path = f'../data/DT/hi/{prune_DT_filename}'
    elif(is_bn):
        DT_edges_path = '../data/DT/bn/bengali_bigram__PruneFeaturesPerWord_1000__FreqSigLMI_s_0_t_0__PruneGraph_p_1000__AggrPerFt__SimCounts1WithFeatures__SimSortlimit_200'
        DT_word_count_path = '../data/DT/bn/bengali_bigram__PruneFeaturesPerWord_1000__WordCount'

        prune_DT_frequency_threshold = 100
        prune_DT_filename = f'prune_{prune_DT_frequency_threshold}_bengali_bigram__PruneFeaturesPerWord_1000__FreqSigLMI_s_0_t_0__PruneGraph_p_1000__AggrPerFt__SimCounts1WithFeatures__SimSortlimit_200'
        prune_DT_edges_path = f'../data/DT/bn/{prune_DT_filename}'
    elif(is_am):
        DT_edges_path = '../data/DT/am/amharic_normalized_dt_bigram_freqSigLMI_dt_edges.txt'
        DT_word_count_path = '../data/DT/am/amharic_normalized_dt_bigram_WordCount.txt'

        prune_DT_frequency_threshold = 100
        prune_DT_filename = f'prune_{prune_DT_frequency_threshold}_amharic_normalized_dt_bigram_freqSigLMI_dt_edges.txt'
        prune_DT_edges_path = f'../data/DT/am/{prune_DT_filename}'
    elif(is_de):
        DT_edges_path = '../data/DT/de_news70M_pruned/LMI_p1000_l_200'
        DT_word_count_path = '../data/DT/de_news70M_pruned/word_count'

        prune_DT_frequency_threshold = 100
        prune_DT_filename = f'prune_{prune_DT_frequency_threshold}_LMI_p1000_l_200'
        prune_DT_edges_path = f'../data/DT/de_news70M_pruned/{prune_DT_filename}'
    update_POS = False

WiC_train_data_path = '../data/WiC/train/train.data.txt'
WiC_train_gold_path = '../data/WiC/train/train.gold.txt'
WiC_dev_data_path = '../data/WiC/dev/dev.data.txt'
WiC_dev_gold_path = '../data/WiC/dev/dev.gold.txt'
WiC_labels = {'T': 0, 'F': 1}

RTE_train_data_path = '../data/glue_data/RTE/train/train.tsv'
RTE_dev_data_path = '../data/glue_data/RTE/dev/dev.tsv'
RTE_labels = {'entailment': 0, 'not_entailment': 1}

STS_B_train_data_path = '../data/glue_data/STS-B/train/sts-train.tsv'
STS_B_dev_data_path = '../data/glue_data/STS-B/dev/sts-dev.tsv'
# does not use labels as this is a regression task
STS_B_columns = ['source', 'type', 'year', 'id', 'score', 'sent_a', 'sent_b']
# use the score_f column to obtain float values

MRPC_train_data_path = '../data/glue_data/MRPC/train/train.tsv'
MRPC_dev_data_path = '../data/glue_data/MRPC/dev/dev.tsv'
MRPC_labels = {'0': 0, '1': 1}
MRPC_columns = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']

SST_2_train_data_path = '../data/glue_data/SST-2/train/train.tsv'
SST_2_dev_data_path = '../data/glue_data/SST-2/dev/dev.tsv'
SST_2_labels = {0: 0, 1: 1}

CoLA_train_data_path = '../data/glue_data/CoLA/train/train.tsv'
CoLA_dev_data_path = '../data/glue_data/CoLA/dev/dev.tsv'
CoLA_labels = {0: 0, 1: 1}

WNLI_translated_train_data_path = '../data/wnli-translated/hi/train/train.csv'
WNLI_translated_dev_data_path = '../data/wnli-translated/hi/dev/dev.csv'
WNLI_translated_labels = {0: 0, 1: 1}

IITP_product_reviews_train_data_path = '../data/iitp-product-reviews/hi/train/hi-train.csv'
IITP_product_reviews_dev_data_path = '../data/iitp-product-reviews/hi/test/hi-test.csv'
IITP_product_reviews_labels = {'positive': 0, 'neutral': 1, 'negative': 2}

MIDAS_discourse_train_json_path = '../data/midas-discourse/hi/train/train.json'
MIDAS_discourse_dev_json_path = '../data/midas-discourse/hi/test/test.json'
MIDAS_discourse_labels = {'Argumentative': 0, 'Descriptive': 1, 'Dialogue': 2, 'Informative': 3, 'Narrative': 4}

DPIL_subtask_1_train_path = '../data/DPIL_csv/subtask_1/train/train_shuffled.csv'
DPIL_subtask_1_dev_path = '../data/DPIL_csv/subtask_1/test/test.csv'
DPIL_subtask_1_labels = {'P': 0, 'NP': 1}

DPIL_subtask_2_train_path = '../data/DPIL_csv/subtask_2/train/train_shuffled.csv'
DPIL_subtask_2_dev_path = '../data/DPIL_csv/subtask_2/test/test.csv'
DPIL_subtask_2_labels = {'P': 0, 'SP': 1, 'NP': 2}

KhondokerIslam_bengali_train_path = '../data/KhondokerIslam_Bengali_Sentiment/train/train.csv'
KhondokerIslam_bengali_dev_path = '../data/KhondokerIslam_Bengali_Sentiment/test/test.csv'
KhondokerIslam_bengali_labels = {0: 0, 1: 1, 2: 2}

rezacsedu_sentiment_train_path = '../data/rezacsedu_sentiment/train/train.csv'
rezacsedu_sentiment_test_path = '../data/rezacsedu_sentiment/test/test.csv'
rezacsedu_sentiment_labels = {'positive': 0, 'negative': 1}

BEmoC_train_path = '../data/BEmoC/train/train.xlsx'
BEmoC_dev_path = '../data/BEmoC/test/test.xlsx'
BEmoC_labels = {'anger': 0, 'fear': 1, 'disgust': 2, 'sadness': 3, 'joy': 4, 'surprise': 5}

Seid_Amharic_Sentiment_train_path = '../data/seid_amharic_sentiment/train/train.txt'
Seid_Amharic_Sentiment_dev_path = '../data/seid_amharic_sentiment/test/test.txt'
Seid_Amharic_Sentiment_labels = {'__label__NEUTRAL': 0, '__label__MIXED': 1, '__label__POSITIVE': 2, '__label__NEGATIVE': 3}
Seid_Amharic_Sentiment_cleaned_labels = {'__label__NEUTRAL': 0, '__label__POSITIVE': 1, '__label__NEGATIVE': 2}

Amharic_eval_path = '../data/sentence_embed_eval/debug.txt'
Amharic_eval_labels = {'__label__NEUTRAL': 0}

germeval2018_train_path = '../data/germeval2018/train/germeval2018.training.txt'
germeval2018_dev_path = '../data/germeval2018/test/germeval2018.test.txt'
germeval2018_subtask_1_labels = {'OFFENSE': 0, 'OTHER': 1}
germeval2018_subtask_2_labels = {'PROFANITY': 0, 'INSULT': 1, 'ABUSE': 2, 'OTHER': 3}

epochs = 3
model_folder = f'../data/models/experiment={experiment}-lm_model_name={lm_model_name}-use_gcn={use_gcn}-use_le={use_le}-exp_no={exp_no}'
if(model_folder[-1] != '/'):
    model_folder += '/'

if(lm_model_name.startswith('bert')):
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 0.01
    max_norm = 1.0
elif(lm_model_name.startswith('roberta') or lm_model_name.startswith('xlm-roberta')):
    beta_1 = 0.9
    beta_2 = 0.98
    weight_decay = 0.1
elif(lm_model_name.startswith('albert')):
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 0.0
    max_norm = 1.0

assert(sum([use_gcn, use_le]) <= 1)
if(is_en):
    assert(experiment in ['WiC', 'RTE', 'STS_B', 'MRPC', 'SST_2', 'CoLA'])
    assert(lm_model_name in ['bert-base-uncased', 'roberta-base'])
else:
    if(is_hi):
        assert(experiment in ['iitp_product', 'midas_discourse', 'wnli_translated', 'dpil_subtask_1', 'dpil_subtask_2'])
        assert(lm_model_name in ['bert-base-multilingual-cased', 'xlm-roberta-base'])
    elif(is_bn):
        assert(experiment in ['KhondokerIslam_bengali', 'rezacsedu_sentiment', 'BEmoC'])
        assert(lm_model_name in ['bert-base-multilingual-cased', 'xlm-roberta-base'])
    elif(is_am):
        assert(experiment in ['seid_amharic_sentiment', 'seid_amharic_cleaned_sentiment'])
        assert(lm_model_name in ['xlm-roberta-base'])
    elif(is_de):
        assert(experiment in ['germeval2018_subtask_1', 'germeval2018_subtask_2'])
        assert(lm_model_name in ['bert-base-multilingual-cased', 'xlm-roberta-base'])

if(train):
    assert(not os.path.exists(model_folder))

model_name = model_folder + 'best_model.pt'
