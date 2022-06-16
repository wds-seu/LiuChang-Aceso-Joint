# -*- coding: utf-8 -*-
# 读取PDF文档 存储到xls文件中

import sys

sys.path.append("..")
from pdf2txt.pdfreader import Grobid
from pdf2txt.pdfreader import PdfReader
from pdf2txt.background.easy_parallelize import *
from pdf2txt.Extractor import Extractor
import time
import os
import hashlib
import pickle
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Run extractFullText.")
    parser.add_argument('--input', nargs='?', default='', help='Input papers path')
    parser.add_argument('--output', nargs='?', default='', help='Output extract results path')
    parser.add_argument('--docs_pkl', nargs='?', default='docs.pkl', help='extractor header pkl file path')
    return parser.parse_args()


def build_dictionary(data_path_list, filepath, file):
    dictionary = []
    filenames = []
    filelength = 0

    args = parse_args()
    file_dir = "input_of_web"
    root = "D:\MYPRO\EBM-Extractor"
    filename = file.filename[0:len(file.filename) - 4]
    save_dir = os.path.join(file_dir, filename)  # 做一个路径存放这个文件相关的内容
    print(filepath, filename)

    data_path = save_dir

    for filename in data_path_list:
        if filename.endswith('.pdf') and filelength < 100:
            filePath = os.path.join(data_path, filename)
            filenames.append(filePath)
            filelength += 1
        if filelength == 100:
            parallelize_head(filenames)
            filelength = 0
            filenames.clear()
    parallelize_head(filenames)

    if not os.path.isfile(args.docs_pkl):
        print('Some errors happened in dictionary building!')
        return None
    else:
        allhead = pickle.load(open(args.docs_pkl, 'rb'))
        dic = [x[0] for x in sorted(dict2list(allhead), key=lambda x: x[1], reverse=True)][:100]

        for head in dic:
            if head.startswith('fig') or head.startswith('table'):
                continue
            elif len(head) > 3:
                dictionary.append(head)
        return dictionary


def loaddic(dicPath):
    dictionary = []
    allhead = pickle.load(open(dicPath, 'rb'))
    dic = [x[0] for x in sorted(dict2list(allhead), key=lambda x: x[1], reverse=True)][:100]

    for head in dic:
        if head.startswith('fig') or head.startswith('table'):
            continue
        elif len(head) > 3:
            dictionary.append(head)
    return dictionary


def parallelize_head(filePath):
    args = parse_args()
    if not os.path.isfile(args.docs_pkl):
        docs = {}
    else:
        docs = pickle.load(open(args.docs_pkl, 'rb'))
    reader = PdfReader()
    reader.connect()
    pdf_binary = []

    for onefile in filePath:
        with open(onefile, 'rb') as files:
            binary_file = files.read()
            pdf_binary.append(binary_file)

    multidic = reader.convert_batch(pdf_binary, num_threads=2)
    extractor = Extractor()
    easy_parallelize(extractor.constractHeadsDictionary, multidic, pool_size=1)
    # dic = [x[0] for x in sorted(dict2list(extractor.current_heads), key=lambda x: x[1], reverse=True)][:50]
    if docs == {}:
        docs = extractor.current_heads
    else:
        for key in extractor.current_heads.keys():
            if key in docs:
                docs[key] += 1
            else:
                docs[key] = 1
    with open(args.docs_pkl, 'wb') as fout:
        pickle.dump(docs, fout)


def parallelize_grobid(filePath, outpath, file_length):
    args = parse_args()
    reader = PdfReader()
    reader.connect()
    pdf_binary = []
    pmid_hash = {}

    for onefile in filePath:
        full_name = os.path.splitext(onefile)
        pmid = full_name[0].rsplit('\\', 1)[-1]
        sha1 = hashlib.sha1()
        with open(onefile, 'rb') as files:
            binary_file = files.read()
            pdf_binary.append(binary_file)
            sha1.update(binary_file)
            pmid_hash[sha1.hexdigest()] = pmid

    multidic = reader.convert_batch(pdf_binary, num_threads=2)
    extractor = Extractor()
    extractor.dictionary_heads = loaddic(args.docs_pkl)

    for i in range(file_length):
        hashid = multidic[i].gold['filehash']
        if hashid in pmid_hash.keys():
            multidic[i].pubmed['pmid'] = pmid_hash[hashid]
            multidic[i].grobid['outpath'] = outpath

    # easy_SerialStart(Extractor().readPDFtoTXT, multidic, file_length)
    easy_parallelize(extractor.readPDFtoTXT, multidic, pool_size=1)


def dict2list(dic):
    # "tranform dict into list"
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def pdf2txt(data_path, out_path):
    """
    抽取PDF文件中的句子，保存为txt文件。
    句子的格式为  sentence id|||section label|||section number|||text
    :param data_path:
    :param out_path:
    :return:
    """
    data_path_list = os.listdir(data_path)
    out_path_list = os.listdir(out_path)
    grobid = Grobid()
    grobid.connect()
    grobid.cleanup()

    filenames = []
    filelength = 0
    for filename in out_path_list:
        if filename.endswith('.txt') or filename.endswith('.xls'):
            filePath = os.path.join(out_path, filename)
            os.remove(filePath)
            print('Delete ' + filename)
    for filename in data_path_list:
        if filename.endswith('.pdf') and filelength < 100:
            filePath = os.path.join(data_path, filename)
            filenames.append(filePath)
            filelength += 1
        if filelength == 100:
            parallelize_grobid(filenames, out_path, filelength)
            filelength = 0
            filenames.clear()
    parallelize_grobid(filenames, out_path, filelength)


def pdf2text(filepath, file):
    args = parse_args()
    # data_path = "G:\\Home\\IBM\\papers"
    # out_path = "G:\\Home\\IBM\\papers\\out"
    # data_path = "/home/penngao/Documents/pdfextract/in6"
    # out_path = "/home/penngao/Documents/pdfextract/out"
    # data_path = args.input
    # out_path = args.output
    file_dir = "input_of_web"
    root = "D:\MYPRO\EBM-Extractor"

    filename = file.filename[0:len(file.filename) - 4]
    save_dir = os.path.join(file_dir, filename)  # 做一个路径存放这个文件相关的内容
    print(filepath, filename)

    data_path = save_dir
    out_path = save_dir
    print(data_path)
    start_time = time.time()
    data_path_list = os.listdir(data_path)
    out_path_list = os.listdir(out_path)
    grobid = Grobid()

    filenames = []
    file_length = 0

    # build dictionary
    head_dictionary = build_dictionary(data_path_list, filepath, file)
    middle_time = time.time()
    print('Time of dictionary construction is: ' + str(round(middle_time - start_time, 3)))

    for filename in out_path_list:
        if filename.endswith('.txt'):
            filePath = os.path.join(out_path, filename)
            os.remove(filePath)
            print('Delete ' + filename)
    for filename in data_path_list:
        #print(filename)
        if filename.endswith('.pdf') and file_length < 100:
            filePath = os.path.join(data_path, filename)
            filenames.append(filePath)
            file_length += 1
        if file_length == 100:
            parallelize_grobid(filenames, out_path, file_length) ,
            file_length = 0
            filenames.clear()
    parallelize_grobid(filenames, out_path, file_length)

    end_time = time.time()
    run_time = end_time - middle_time
    print('Time of processing is: ' + str(round(run_time, 3)))
