
import numpy as np
import re, os, json
from random import choice

BERT_MAX_LEN = 512
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


def load_data(train_path, dev_path, test_path, rel_dict_path):  
    train_data = json.load(open(train_path, encoding="utf-8"))  
    dev_data = json.load(open(dev_path, encoding="utf-8"))
    test_data = json.load(open(test_path))
    id2rel, rel2id = json.load(open(rel_dict_path, encoding="utf-8"))

    id2rel = {int(i): j for i, j in id2rel.items()}  
    num_rels = len(id2rel)  

    random_order = list(range(len(train_data)))  
    np.random.seed(RANDOM_SEED)  
    np.random.shuffle(random_order)  
    train_data = [train_data[i] for i in random_order]  

    print("train_data len:", len(train_data))  
    print("dev_data len:", len(dev_data))
    print("test_data len:", len(test_data))

    return train_data, dev_data, test_data, id2rel, rel2id, num_rels  


class data_generator:  
    def __init__(self, data, tokenizer, rel2id, num_rels, maxlen,
                 batch_size=32):  
        self.data = data  
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.rel2id = rel2id
        self.num_rels = num_rels
        self.steps = len(self.data) // self.batch_size  
        if len(self.data) % self.batch_size != 0:  
            self.steps += 1

    def __len__(self):  
        return self.steps

    def __iter__(self):
        while True:  
            idxs = list(range(len(self.data)))  
            np.random.seed(RANDOM_SEED)  
            np.random.shuffle(idxs)
            tokens_batch, segments_batch, sub_type_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], []
            for idx in idxs:  
                line = self.data[
                    idx]  
                text = ' '.join(line['text'].split()[
                                :self.maxlen])  
                tokens = self.tokenizer.tokenize(
                    text)  
                if len(tokens) > BERT_MAX_LEN:  
                    tokens = tokens[:BERT_MAX_LEN]  
                text_len = len(tokens)  

                label = line['label']

                if label != '':  
                    token_ids, segment_ids = self.tokenizer.encode(first=text)  
                    if len(token_ids) > text_len:  
                        token_ids = token_ids[
                                    :text_len]  
                        segment_ids = segment_ids[
                                      :text_len]  
                    tokens_batch.append(token_ids)  
                    segments_batch.append(segment_ids)  
                    sub_type = np.zeros(4)  

                    s = label
                    if s == 'P':
                        sub_type[0] = 1
                    elif s == 'I':
                        sub_type[1] = 1
                    elif s == 'O':
                        sub_type[2] = 1
                    elif s =='N':
                        sub_type[3] = 1
                    

                    sub_type_batch.append(sub_type)  

                    if len(tokens_batch) == self.batch_size or idx == idxs[-1]:  
                        tokens_batch = seq_padding(tokens_batch)  
                        segments_batch = seq_padding(segments_batch)
                        sub_type_batch = np.array(sub_type_batch)

                        yield [tokens_batch, segments_batch, sub_type_batch], None
                        tokens_batch, segments_batch, sub_type_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], []  
