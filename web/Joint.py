#! -*- coding:utf-8 -*-
from models.data_loader_je import data_generator, load_data
from models.model_je import E2EModel, Evaluate
from models.utils_je import get_tokenizer, metric
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


def load_je_model():
    # pre-trained bert model config
    bert_model = 'biobert_v1.1_pubmed'
    bert_config_path = 'pretrained_bio_bert_models/' + bert_model + '/bert_config.json'
    bert_vocab_path = 'pretrained_bio_bert_models/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = 'pretrained_bio_bert_models/' + bert_model + '/model.ckpt-1000000'
    models = []
    for dataset in ['EBM-NLP-A', 'EBM-NLP-R']:
        model = []
        with graph.as_default():
            save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'
            LR = 1e-5
            if dataset == 'EBM-NLP-A':
                num_rels = 5
            elif dataset == 'EBM-NLP-R':
                num_rels = 11
            # 获取数据
            subject_model, object_model, ebm_model = E2EModel(bert_config_path, bert_checkpoint_path, LR,
                                                              num_rels)  # 获取模型

            ebm_model.load_weights(save_weights_path)
            model.append(subject_model)
            model.append(object_model)
        models.append(model)
    return models


def extract_medical_triples_1(data_Joint, result_path, models):
    # pre-trained bert model config
    bert_model = 'biobert_v1.1_pubmed'
    bert_vocab_path = 'pretrained_bio_bert_models/' + bert_model + '/vocab.txt'
    result_paths = []
    for dataset in ['EBM-NLP-A', 'EBM-NLP-R']:
        i = 0
        with graph.as_default():
            train_path = 'data/' + dataset + '/train_triples.json'
            dev_path = 'data/' + dataset + '/dev_triples.json'
            test_path = data_Joint
            rel_dict_path = 'data/' + dataset + '/rel2id.json'
            sub_text_path = 'data/' + dataset + '/sub2text.json'

            tokenizer = get_tokenizer(bert_vocab_path)  # 定义tokenizer并初始化
            train_data, dev_data, test_data, id2rel, rel2id, num_rels, sub2text, sub_labels = load_data(train_path,
                                                                                                        dev_path,
                                                                                                        test_path,
                                                                                                        rel_dict_path,
                                                                                                        sub_text_path)  # 获取数据

            a_r = dataset.split('-')[-1]
            test_result_path = os.path.join(result_path, 'Joint_' + a_r + '_result.json')
            result_paths.append(test_result_path)
            isExactMatch = True
            precision, recall, f1_score = metric(models[i][0], models[i][1], test_data, id2rel, tokenizer, sub2text,
                                                 sub_labels, isExactMatch,
                                                 test_result_path)
            print(f'{precision}\t{recall}\t{f1_score}')
        i += 1
    return result_paths


def extract_medical_triples(data_Joint, result_path):
    # pre-trained bert model config
    bert_model = 'biobert_v1.1_pubmed'
    bert_config_path = 'pretrained_bio_bert_models/' + bert_model + '/bert_config.json'
    bert_vocab_path = 'pretrained_bio_bert_models/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = 'pretrained_bio_bert_models/' + bert_model + '/model.ckpt-1000000'
    result_paths = []
    for dataset in ['EBM-NLP-A', 'EBM-NLP-R']:
        with graph.as_default():
            train_path = 'data/' + dataset + '/train_triples.json'
            dev_path = 'data/' + dataset + '/dev_triples.json'
            test_path = data_Joint
            rel_dict_path = 'data/' + dataset + '/rel2id.json'
            sub_text_path = 'data/' + dataset + '/sub2text.json'
            save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'

            LR = 1e-5
            tokenizer = get_tokenizer(bert_vocab_path)  # 定义tokenizer并初始化
            train_data, dev_data, test_data, id2rel, rel2id, num_rels, sub2text, sub_labels = load_data(train_path,
                                                                                                        dev_path,
                                                                                                        test_path,
                                                                                                        rel_dict_path,
                                                                                                        sub_text_path)  # 获取数据
            subject_model, object_model, ebm_model = E2EModel(bert_config_path, bert_checkpoint_path, LR,
                                                              num_rels)  # 获取模型

            ebm_model.load_weights(save_weights_path)
            a_r = dataset.split('-')[-1]
            test_result_path = os.path.join(result_path, 'Joint_' + a_r + '_result.json')
            result_paths.append(test_result_path)
            isExactMatch = True
            precision, recall, f1_score = metric(subject_model, object_model, test_data, id2rel, tokenizer, sub2text,
                                                 sub_labels, isExactMatch,
                                                 test_result_path)
            print(f'{precision}\t{recall}\t{f1_score}')
    return result_paths


def joint_result_collation(result_files, result_path):
    print('joint result collation')
    result_data = []
    p_data = []
    i_data = []
    o_data = []
    for file in result_files:
        with open(file, encoding="utf-8") as f:
            str = ''
            for l in tqdm(f):
                str += l
                if l == '}\n':
                    str.replace('\n', '').replace('    ', '') + '\n'
                    # print(str)

                    a = json.loads(str)
                    pred_triples = a['triple_list_pred']

                    for t in pred_triples:
                        if file.split('_')[-2] == 'A':
                            attribute = {'subject': t['subject'], 'attribute': t['relation'], 'value': t['object']}
                            if t['subject'] == 'P':
                                p_data.append(attribute)
                            elif t['subject'] == 'I':
                                i_data.append(attribute)
                            elif t['subject'] == 'O':
                                o_data.append(attribute)
                        elif file.split('_')[-2] == 'R':
                            if t['subject'] == 'P':
                                p_data.append(t)
                            elif t['subject'] == 'I':
                                i_data.append(t)
                            elif t['subject'] == 'O':
                                o_data.append(t)
                    str = ''

    # 去重
    p_data = deleteDuplicate(p_data)
    i_data = deleteDuplicate(i_data)
    o_data = deleteDuplicate(o_data)

    data_line_p = {
        'label': 'P',
        'pred_triples': p_data
    }
    data_line_i = {
        'label': 'I',
        'pred_triples': i_data
    }
    data_line_o = {
        'label': 'O',
        'pred_triples': o_data
    }
    result_data.append(data_line_p)
    result_data.append(data_line_i)
    result_data.append(data_line_o)

    result_collection_path = os.path.join(result_path, 'Joint_result_collation.json')
    with open(os.path.join(result_path, 'Joint_result_collation.json'), "a+") as out_file:
        json.dump(result_data, out_file, indent=4, ensure_ascii=False)

    return json.dumps(result_data), result_collection_path


def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        try:
            initializer = next(it)
        except StopIteration:
            raise TypeError('reduce() of empty sequence with no initial value')
    accum_value = initializer
    for x in it:
        accum_value = function(accum_value, x)
    return accum_value


def deleteDuplicate(li):
    func = lambda x, y: x if y in x else x + [y]
    li = reduce(func, [[], ] + li)
    return li


# result_files = [
#     'D:\MYPRO\EBM-Extractor\input_of_web\Cerebral Venous Thrombosis as a Complication of\Joint_A_result.json',
#     'D:\MYPRO\EBM-Extractor\input_of_web\Cerebral Venous Thrombosis as a Complication of\Joint_R_result.json']
# result_path = 'D:\MYPRO\EBM-Extractor\input_of_web\Cerebral Venous Thrombosis as a Complication of'
# joint_result_collation(result_files, result_path)
