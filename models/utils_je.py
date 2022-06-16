import keras.backend as K
from keras_bert import Tokenizer
import numpy as np
import codecs
from tqdm import tqdm
import json
import unicodedata
from models.data_loader_je import sub_encode
import tensorflow as tf

BERT_MAX_LEN = 128


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
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)


def extract_items(subject_model, object_model, tokenizer, text_in, id2rel, sub2text, sub_labels, h_bar=0.5, t_bar=0.5):
    sub_token_ids, sub_segment_ids, sub_max_length = sub_encode(sub2text, tokenizer)

    tokens = tokenizer.tokenize(text_in)
    token_ids, segment_ids = tokenizer.encode(first=text_in)
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:, :BERT_MAX_LEN]
        segment_ids = segment_ids[:, :BERT_MAX_LEN]

    sub_type_logits = subject_model.predict([token_ids, segment_ids])

    sub_type_index = np.argmax(sub_type_logits)

    triple_list = []
    pred_sub = ''

    sub_type = np.zeros(3)
    if sub_type_index == 0:

        pred_sub = 'P'
        sub_type[0] = 1
    elif sub_type_index == 1:

        pred_sub = 'I'
        sub_type[1] = 1
    elif sub_type_index == 2:

        pred_sub = 'O'
        sub_type[2] = 1

    sub_type_tokens_ids = sub_token_ids[sub_type_index]
    sub_type_segment_ids = sub_segment_ids[sub_type_index]

    token_ids = np.repeat(token_ids, 1, 0)
    segment_ids = np.repeat(segment_ids, 1, 0)
    sub_head_idx = np.array(np.ones((1, 1)))
    sub_tail_idx = np.reshape(np.repeat(np.array([sub_max_length - 3]), 1), (1, 1))
    sub_type_tokens_ids = np.array(sub_type_tokens_ids)
    sub_type_segment_ids = np.array(sub_type_segment_ids)

    obj_heads_logits, obj_tails_logits = object_model.predict(
        [token_ids, segment_ids, sub_type, sub_head_idx, sub_tail_idx, sub_type_tokens_ids, sub_type_segment_ids])

    obj_heads, obj_tails = np.where(obj_heads_logits[0] > h_bar), np.where(obj_tails_logits[0] > t_bar)
    for obj_head, rel_head in zip(*obj_heads):
        for obj_tail, rel_tail in zip(*obj_tails):
            if obj_head <= obj_tail and rel_head == rel_tail:
                rel = id2rel[rel_head]
                sub = rel.split("/")[1]
                obj = tokens[obj_head: obj_tail]
                obj = ''.join([i.lstrip("##") for i in obj])
                obj = ' '.join(obj.split('[unused1]'))
                triple_list.append((sub, rel, obj))
                break

    if triple_list:
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set), pred_sub
    else:
        return [], pred_sub


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def sub_rel_obj_match(triple_set):
    sub_set = {i[0] for i in triple_set}
    rel_set = {i[1] for i in triple_set}
    obj_set = {i[2] for i in triple_set}
    return sub_set, rel_set, obj_set


def f1_precision_recall(predict_num, gold_num, correct_num):
    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score, precision, recall


def pio_triples(Pred_triples):
    p_triples, i_triples, o_triples = [], [], []
    for triple in Pred_triples:
        sub = triple[0]
        if sub == 'P':
            p_triples.append(triple)
        elif sub == 'I':
            i_triples.append(triple)
        elif sub == 'O':
            o_triples.append(triple)
    p_triples = set(p_triples)
    i_triples = set(i_triples)
    o_triples = set(o_triples)
    return p_triples, i_triples, o_triples


def metric(subject_model, object_model, eval_data, id2rel, tokenizer, sub2text, sub_labels, exact_match=False,
           output_path=None):
    if output_path:
        F = open(output_path, 'w', encoding="utf-8")
    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10

    sub_type_correct_num, sub_type_predict_num = 1e-10, 1e-10
    p_correct_num, p_predict_num, p_gold_num, p_sub_correct_num, p_rel_correct_num, p_obj_correct_num = 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10
    i_correct_num, i_predict_num, i_gold_num, i_sub_correct_num, i_rel_correct_num, i_obj_correct_num = 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10
    o_correct_num, o_predict_num, o_gold_num, o_sub_correct_num, o_rel_correct_num, o_obj_correct_num = 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        Pred_triples, pred_sub = extract_items(subject_model, object_model, tokenizer, line['text'], id2rel, sub2text,
                                               sub_labels)
        Pred_triples = set(Pred_triples)
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (
            Pred_triples, Gold_triples)

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score
