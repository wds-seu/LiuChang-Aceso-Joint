#! -*- coding:utf-8 -*-
import keras.backend as K
from keras_bert import Tokenizer
import numpy as np
import codecs
from tqdm import tqdm
import json
import unicodedata

BERT_MAX_LEN = 256


class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append('[unused1]')
        return tokens


def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)


def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')  # 执行 tensorflow 中的张量数据类型转换
    batch_idxs = K.arange(0, K.shape(seq)[0])  # 生成[0,1,2,...,seq.shape[0]] 的tensor
    batch_idxs = K.expand_dims(batch_idxs, 1)  # 在下标为1的轴上增加一维 :shape= (seq.shape[0], 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)  # 所有输入张量通过 1 轴串联起来的输出张量
    return K.tf.gather_nd(seq, idxs)


def extract_items(subject_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):  # 提取元素
    tokens = tokenizer.tokenize(text_in)  # 原句分词
    token_ids, segment_ids = tokenizer.encode(first=text_in)  # token_ids, segment_ids
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])  # token_ids, segment_ids转换为向量
    if len(token_ids[0]) > BERT_MAX_LEN:  # 截取长度
        token_ids = token_ids[:, :BERT_MAX_LEN]
        segment_ids = segment_ids[:, :BERT_MAX_LEN]
        # sub_heads_logits, sub_tails_logits = subject_model.predict([token_ids, segment_ids])
    # sub_heads, sub_tails = np.where(sub_heads_logits[0] > h_bar)[0], np.where(sub_tails_logits[0] > t_bar)[0]
    sub_type_logits = subject_model.predict([token_ids, segment_ids])
    sub_type_index = np.argmax(sub_type_logits)

    pred_label = ''

    # for sub_head in sub_heads:
    # sub_tail = sub_tails[sub_tails >= sub_head]
    # if len(sub_tail) > 0:
    # sub_tail = sub_tail[0]
    # subject = tokens[sub_head: sub_tail]
    # subjects.append((subject, sub_head, sub_tail))

    if sub_type_index == 0:
        pred_label = 'P'
    elif sub_type_index == 1:
        pred_label = 'I'
    elif sub_type_index == 2:
        pred_label = 'O'
    elif sub_type_index == 3:
        pred_label = 'N'

    # if subjects:
    #     triple_list = []
    #     token_ids = np.repeat(token_ids, len(subjects), 0)
    #     segment_ids = np.repeat(segment_ids, len(subjects), 0)
    #     # sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
    #     obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids])
    #     for i, subject in enumerate(subjects):
    #         # sub = subject[0]
    #         # sub = ''.join([i.lstrip("##") for i in sub])
    #         # sub = ' '.join(sub.split('[unused1]'))
    #         sub = subject
    #         obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
    #         for obj_head, rel_head in zip(*obj_heads):
    #             for obj_tail, rel_tail in zip(*obj_tails):
    #                 if obj_head <= obj_tail and rel_head == rel_tail:
    #                     rel = id2rel[rel_head]
    #                     obj = tokens[obj_head: obj_tail]
    #                     obj = ''.join([i.lstrip("##") for i in obj])
    #                     obj = ' '.join(obj.split('[unused1]'))
    #                     triple_list.append((sub, rel, obj))
    #                     break
    #     triple_set = set()
    #     for s, r, o in triple_list:
    #         triple_set.add((s, r, o))
    #     return list(triple_set)
    # else:
    #     return []
    return pred_label


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def sub_rel_obj_match(gold_set):
    gold_sub = [i[0] for i in gold_set]
    gold_rel = [i[1] for i in gold_set]
    gold_obj = [i[2] for i in gold_set]
    return gold_sub, gold_rel, gold_obj


def f1_precision_recall(predict_num, gold_num, correct_num):
    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score, precision, recall


def metric(subject_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    if output_path:
        F = open(output_path, 'w', encoding="utf-8")
    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10

    sub_type_correct_num, sub_type_predict_num = 1e-10, 1e-10
    p_correct_num, p_predict_num, p_gold_num = 1e-10, 1e-10, 1e-10
    i_correct_num, i_predict_num, i_gold_num = 1e-10, 1e-10, 1e-10
    o_correct_num, o_predict_num, o_gold_num = 1e-10, 1e-10, 1e-10
    n_correct_num, n_predict_num, n_gold_num = 1e-10, 1e-10, 1e-10
    data = []
    for line in tqdm(iter(eval_data)):
        Pred_label = extract_items(subject_model, tokenizer, line['text'], id2rel)
        Gold_label = line['label']

        if Pred_label == Gold_label:
            correct_num += 1
        predict_num += 1
        gold_num += 1

        if Gold_label == 'P':
            p_gold_num += 1
            if Pred_label == Gold_label:
                p_correct_num += 1

        elif Gold_label == 'I':
            i_gold_num += 1
            if Pred_label == Gold_label:
                i_correct_num += 1

        elif Gold_label == 'O':
            o_gold_num += 1
            if Pred_label == Gold_label:
                o_correct_num += 1

        elif Gold_label == 'N':
            n_gold_num += 1
            if Pred_label == Gold_label:
                n_correct_num += 1

        if Pred_label == 'P':
            p_predict_num += 1

        elif Pred_label == 'I':
            i_predict_num += 1

        elif Pred_label == 'O':
            o_predict_num += 1

        elif Pred_label == 'N':
            n_predict_num += 1

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'label_gold': Gold_label,
                'label_pred': Pred_label
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        # result = json.dumps(data, indent=4, ensure_ascii=False)
        # F.write(result)
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    # 其他指标计算
    p_f1_score, p_precision, p_recall = f1_precision_recall(p_predict_num, p_gold_num, p_correct_num)
    print('-----------------------------------P-----------------------------------')
    print(f'p_correct_num:{p_correct_num}\np_predict_num:{p_predict_num}\np_gold_num:{p_gold_num}')
    print('p_f1: %.4f, p_precision: %.4f, p_recall: %.4f\n' % (
        p_f1_score, p_precision, p_recall))

    i_f1_score, i_precision, i_recall = f1_precision_recall(i_predict_num, i_gold_num, i_correct_num)
    print('-----------------------------------I-----------------------------------')
    print(f'i_correct_num:{i_correct_num}\ni_predict_num:{i_predict_num}\ni_gold_num:{i_gold_num}')
    print('i_f1: %.4f, i_precision: %.4f, i_recall: %.4f\n' % (
        i_f1_score, i_precision, i_recall))

    o_f1_score, o_precision, o_recall = f1_precision_recall(o_predict_num, o_gold_num, o_correct_num)
    print('-----------------------------------O-----------------------------------')
    print(f'o_correct_num:{o_correct_num}\no_predict_num:{o_predict_num}\no_gold_num:{o_gold_num}')
    print('o_f1: %.4f, o_precision: %.4f, o_recall: %.4f\n' % (
        o_f1_score, o_precision, o_recall))

    n_f1_score, n_precision, n_recall = f1_precision_recall(n_predict_num, n_gold_num, n_correct_num)
    print('-----------------------------------N-----------------------------------')
    print(f'n_correct_num:{n_correct_num}\nn_predict_num:{n_predict_num}\nn_gold_num:{n_gold_num}')
    print('n_f1: %.4f, n_precision: %.4f, n_recall: %.4f\n' % (
        n_f1_score, n_precision, n_recall))

    print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
    return precision, recall, f1_score
