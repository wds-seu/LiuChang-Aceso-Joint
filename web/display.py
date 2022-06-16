#! -*- coding:utf-8 -*-
import os, argparse
import json
from tqdm import tqdm

def seme_result_display(result_collection_file):
    print('seme result display')

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
