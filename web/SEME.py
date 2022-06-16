#! -*- coding:utf-8 -*-
from models.data_loader_se import data_generator, load_data
from models.model_se import E2EModel, Evaluate
from models.utils_se import get_tokenizer, metric
import os, argparse
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras import backend as K

if (K.backend() == 'tensorflow'):
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 最大显存占用率
    sess = tf.Session(config=config)

global graph
graph = tf.get_default_graph()

parser = argparse.ArgumentParser(description='Model Controller')
args = parser.parse_args()

def load_se_model():
    bert_model = 'biobert_v1.1_pubmed'
    bert_config_path = 'pretrained_bio_bert_models/' + bert_model + '/bert_config.json'
    bert_checkpoint_path = 'pretrained_bio_bert_models/' + bert_model + '/model.ckpt-1000000'
    with graph.as_default():
        # pre-trained bert model config
        dataset = 'PION-DS-LQ'

        save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'
        LR = 1e-5
        num_rels = 0
        sub_model, subject_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)  # 获取模型

        sub_model.load_weights(save_weights_path)

    return subject_model

def extract_medical_summary_1(data_SEME, result_path, subject_model):
    # pre-trained bert model config
    bert_model = 'biobert_v1.1_pubmed'
    bert_vocab_path = 'pretrained_bio_bert_models/' + bert_model + '/vocab.txt'
    with graph.as_default():
        # pre-trained bert model config
        dataset = 'PION-DS-LQ'
        train_path = 'data/' + dataset + '/train_triples.json'
        dev_path = 'data/' + dataset + '/dev_triples.json'
        test_path = data_SEME
        rel_dict_path = 'data/' + dataset + '/rel2id.json'

        tokenizer = get_tokenizer(bert_vocab_path)  # 定义tokenizer并初始化
        train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                              rel_dict_path)  # 获取数据

        test_result_path = os.path.join(result_path, 'SEME_result.json')
        isExactMatch = True
        precision, recall, f1_score = metric(subject_model, test_data, id2rel, tokenizer, isExactMatch,
                                             test_result_path)
        print(f'{precision}\t{recall}\t{f1_score}')

    return test_result_path

def extract_medical_summary(data_SEME, result_path):
    # pre-trained bert model config
    bert_model = 'biobert_v1.1_pubmed'
    bert_config_path = 'pretrained_bio_bert_models/' + bert_model + '/bert_config.json'
    bert_vocab_path = 'pretrained_bio_bert_models/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = 'pretrained_bio_bert_models/' + bert_model + '/model.ckpt-1000000'
    with graph.as_default():
        # pre-trained bert model config
        dataset = 'PION-DS-LQ'
        train_path = 'data/' + dataset + '/train_triples.json'
        dev_path = 'data/' + dataset + '/dev_triples.json'
        test_path = data_SEME
        rel_dict_path = 'data/' + dataset + '/rel2id.json'
        save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'
        LR = 1e-5

        tokenizer = get_tokenizer(bert_vocab_path)  # 定义tokenizer并初始化
        train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                              rel_dict_path)  # 获取数据
        sub_model, subject_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)  # 获取模型

        sub_model.load_weights(save_weights_path)
        test_result_path = os.path.join(result_path, 'SEME_result.json')
        isExactMatch = True
        precision, recall, f1_score = metric(subject_model, test_data, id2rel, tokenizer, isExactMatch,
                                             test_result_path)
        print(f'{precision}\t{recall}\t{f1_score}')

    return test_result_path


def seme_result_collation(result_file, result_path):
    print('seme result collation')
    result_data = []
    with open(result_file, encoding="utf-8") as f:
        p_data = []
        i_data = []
        o_data = []
        i = 0
        str = ''
        for l in tqdm(f):
            str += l
            i += 1
            if i % 5 == 0:
                str.replace('\n', '').replace('    ', '') + '\n'
                print(str)

                a = json.loads(str)
                pred_label = a['label_pred']
                text = a['text']

                if pred_label == 'P':
                    p_data.append(text)
                elif pred_label == 'I':
                    i_data.append(text)
                elif pred_label == 'O':
                    o_data.append(text)

                i = 0
                str = ''

        data_line_p = {
            'label': 'P',
            'text': p_data
        }
        data_line_i = {
            'label': 'I',
            'text': i_data
        }
        data_line_o = {
            'label': 'O',
            'text': o_data
        }
        result_data.append(data_line_p)
        result_data.append(data_line_i)
        result_data.append(data_line_o)

    result_collection_path = os.path.join(result_path, 'SEME_result_collation.json')
    with open(os.path.join(result_path, 'SEME_result_collation.json'), "a+") as out_file:
        json.dump(result_data, out_file, indent=4, ensure_ascii=False)

    return json.dumps(result_data), result_collection_path
