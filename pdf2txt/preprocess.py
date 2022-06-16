import json
import os
from tqdm import tqdm
import csv
import random

def preprosess_seme(text, filename, save_dir):
    data = []
    with open(text, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            text = line.split('|||')[-1]
            label= 'X'
            print(label, text)
            data_line = {
                'text': text,
                'label': label
            }
            data.append(data_line)


    with open(os.path.join(save_dir, 'SEME_' + filename) + '.json', "a+") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)

    return os.path.join(save_dir, 'SEME_' + filename) + '.json'

def preprosess_joint(seme_result, filename, save_dir):
    data = []
    with open(seme_result, encoding="utf-8") as f:
        i = 0
        str = ''
        for l in tqdm(f):
            str += l
            i += 1
            if i % 5 == 0:
                str.replace('\n', '').replace('    ', '') + '\n'
                a = json.loads(str)
                pred_label = a['label_pred']
                text = a['text']
                print(text)
                data_line = {
                    'text': text,
                    'triple_list': []
                }
                data.append(data_line)

                i = 0
                str = ''

    with open(os.path.join(save_dir, 'Joint_' + filename) + '.json', "a+") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)

    return os.path.join(save_dir, 'Joint_' + filename) + '.json'





