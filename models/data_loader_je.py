import numpy as np
import re, os, json
from random import choice
from keras_bert import Tokenizer

BERT_MAX_LEN = 128
RANDOM_SEED = 2022


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list


def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])


def sub_encode(sub2text, tokenizer):
    sub_token_ids_batch, sub_segment_ids_batch = [], []
    sub_token_ids_out, sub_segment_ids_out = [], []
    for sub in sub2text:
        sub_text = sub2text[sub]
        sub_token_ids, sub_segment_ids = tokenizer.encode(first=sub_text)
        sub_token_ids = sub_token_ids
        sub_segment_ids = sub_segment_ids
        sub_token_ids_batch.append(sub_token_ids)
        sub_segment_ids_batch.append(sub_segment_ids)

    length_batch = [len(seq) for seq in sub_token_ids_batch]
    max_length = max(length_batch)

    for seq in sub_token_ids_batch:
        seq = np.array(np.concatenate([seq, [0] * (max_length - len(seq))]) if len(seq) < max_length else seq)
        sub_token_ids_out.append(seq)

    for seq in sub_segment_ids_batch:
        seq = np.array(np.concatenate([seq, [0] * (max_length - len(seq))]) if len(seq) < max_length else seq)
        sub_segment_ids_out.append(seq)

    return sub_token_ids_out, sub_segment_ids_out, max_length


def load_data(train_path, dev_path, test_path, rel_dict_path,
              sub_text_path):
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2rel, rel2id = json.load(open(rel_dict_path))
    sub2text = json.load(open(sub_text_path))

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    random_order = list(range(len(train_data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(random_order)
    train_data = [train_data[i] for i in random_order]

    for sent in train_data:
        to_tuple(sent)
    for sent in dev_data:
        to_tuple(sent)
    for sent in test_data:
        to_tuple(sent)

    p_labels, i_labels, o_labels, sub_labels = [], [], [], []
    rel_labels = rel2id
    for rel_label in rel_labels:
        rel_sub = rel_label.split("/")[1]
        if rel_sub == 'P':
            p_labels.append(rel_label)
        elif rel_sub == 'I':
            i_labels.append(rel_label)
        elif rel_sub == 'O':
            o_labels.append(rel_label)

    sub_labels.append(p_labels)
    sub_labels.append(i_labels)
    sub_labels.append(o_labels)

    print("train_data len:", len(train_data))
    print("dev_data len:", len(dev_data))
    print("test_data len:", len(test_data))

    return train_data, dev_data, test_data, id2rel, rel2id, num_rels, sub2text, sub_labels


class data_generator:
    def __init__(self, data, tokenizer, rel2id, num_rels, sub2text, sub_labels, maxlen,
                 batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.rel2id = rel2id
        self.num_rels = num_rels
        self.sub2text = sub2text
        self.sub_labels = sub_labels
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)

            tokens_batch, segments_batch = [], []
            gold_sub_type_batch, sub_head_idx_batch, sub_tail_idx_batch = [], [], []

            sub_token_batch, sub_segment_batch = [], []
            obj_heads_batch, obj_tails_batch = [], []

            sub_token_ids, sub_segment_ids, sub_max_length = sub_encode(self.sub2text, self.tokenizer)

            p_labels, i_labels, o_labels = self.sub_labels

            for idx in idxs:
                line = self.data[idx]
                text = ' '.join(line['text'].split()[:self.maxlen])
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > BERT_MAX_LEN:
                    tokens = tokens[:BERT_MAX_LEN]
                text_len = len(tokens)

                s2ro_map = {}
                for triple in line['triple_list']:
                    triple = (
                        self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                    sub = triple[0][0]
                    obj_head_idx = find_head_idx(tokens, triple[2])
                    if obj_head_idx != -1:
                        if sub not in s2ro_map:
                            s2ro_map[sub] = []
                        s2ro_map[sub].append((obj_head_idx,
                                              obj_head_idx + len(triple[2]) - 1,
                                              self.rel2id[triple[1]]))

                if s2ro_map:
                    token_ids, segment_ids = self.tokenizer.encode(first=text)
                    if len(token_ids) > text_len:
                        token_ids = token_ids[
                                    :text_len]
                        segment_ids = segment_ids[
                                      :text_len]
                    tokens_batch.append(token_ids)
                    segments_batch.append(segment_ids)
                    sub_type = np.zeros(3)
                    for s in s2ro_map:
                        if s == 'P':
                            sub_type[0] = 1
                            sub_token_batch.append(sub_token_ids[0])
                            sub_segment_batch.append(sub_segment_ids[0])
                        elif s == 'I':
                            sub_type[1] = 1
                            sub_token_batch.append(sub_token_ids[1])
                            sub_segment_batch.append(sub_segment_ids[1])
                        elif s == 'O':
                            sub_type[2] = 1
                            sub_token_batch.append(sub_token_ids[2])
                            sub_segment_batch.append(sub_segment_ids[2])

                    s = choice(list(s2ro_map.keys()))
                    obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros(
                        (text_len, self.num_rels))
                    for ro in s2ro_map.get(s, []):
                        obj_heads[ro[0]][ro[2]] = 1
                        obj_tails[ro[1]][ro[2]] = 1

                    gold_sub_type_batch.append(sub_type)
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)

                    if len(tokens_batch) == self.batch_size or idx == idxs[-1]:
                        tokens_batch = seq_padding(tokens_batch)
                        segments_batch = seq_padding(segments_batch)

                        gold_sub_type_batch = seq_padding(gold_sub_type_batch)
                        sub_token_batch = seq_padding(sub_token_batch)
                        sub_segment_batch = seq_padding(sub_segment_batch)

                        obj_heads_batch = seq_padding(obj_heads_batch, np.zeros(self.num_rels))
                        obj_tails_batch = seq_padding(obj_tails_batch, np.zeros(self.num_rels))

                        sub_head_idx_batch = np.array(np.ones((tokens_batch.shape[0], 1)))
                        sub_tail_idx_batch = np.reshape(
                            np.repeat(np.array([sub_max_length - 3]), tokens_batch.shape[0]),
                            (tokens_batch.shape[0], 1))

                        yield [tokens_batch, segments_batch, gold_sub_type_batch, sub_head_idx_batch,
                               sub_tail_idx_batch,
                               sub_token_batch, sub_segment_batch, obj_heads_batch, obj_tails_batch], None

                        tokens_batch, segments_batch = [], []
                        gold_sub_type_batch, sub_head_idx_batch, sub_tail_idx_batch = [], [], []

                        sub_token_batch, sub_segment_batch = [], []
                        obj_heads_batch, obj_tails_batch = [], []
