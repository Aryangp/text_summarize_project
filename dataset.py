import glob
import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm
sys.path.insert(0, '../')
from utilities import *
from sklearn.metrics.pairwise import cosine_similarity
import torch



def get_root_path():
    '''
    function to get root path of dataset

    change the path variable to the path of the dataset
    '''
    path = "C:/Users/gupta/OneDrive/Desktop/code/python-projects/text-sum-final"
    return path


def get_summary_data(dataset, train):
    '''
    function to get names, documents, and summaries

    change the path variable to the path of the dataset
    '''
    if dataset == "N2":
        path = get_root_path() + '/N2/Full-Text/India'
        all_files = glob.glob(path + "/*.txt")

        data_source = []
        names = []
        for filename in all_files:
            with open(filename, 'r') as f: 
                p = filename.rfind("/")
#                 print(filename[p+1:])
                names.append(filename[p+1:])
                a = f.read()
                data_source.append(a)
        return names, data_source, []
    
    path = get_root_path() + "/data/judgement"
    all_files = glob.glob(path + "/*.txt")
    data_source = []
    names = []
    for filename in all_files:
        with open(filename, 'r') as f:
            p = filename.rfind("/")
            names.append(filename[p+1:])
            a = f.read()
            data_source.append(a)
    path1 = get_root_path() + '/data/summary/A1'
    all_files = glob.glob(path1 + "/*.txt")
    data_summary = []
    for filename in all_files:
        with open(filename, 'r') as f: 
            a = f.read()
            l = len(a)
            data_summary.append(a)
            
    return names, data_source, data_summary


dataset="IN"

names, data_source, data_summary = get_summary_data(dataset, "train")

print(len(names), len(data_source), len(data_summary))


# Now once we get the data now have to prepare the data
from transformers import AutoTokenizer
from transformers import  BertModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = False).to("cuda")
bert_model.eval()

def get_sen_encoding(sents):
    '''
    Function to generate encoding for each word in the input list 
    input: sents - List of sentences
    returns the list of the sentence encoding 
    '''
    a = 0.001
    answer = None
    for sent in sents:
        ip =tokenizer(sent, return_tensors='pt', max_length=150, truncation=True, padding='max_length')
        tokens = tokenizer.convert_ids_to_tokens(ip['input_ids'][0])
        ip = ip.to("cuda")
        bert_output = bert_model(**ip)
        embedding = bert_output['pooler_output'].clone().detach()
        embedding = embedding.to("cpu")
        if answer == None:
            answer = embedding
            answer.resize_(1, 768)
        else:
            embedding.resize_(1, 768)
            answer = torch.cat((answer, embedding),0)
    return answer


def similarity_l_l(l1, l2):
    '''
    Function to find the most similar sentence in the document for each sentence in the summary 
    input:  l1 - Summary sentences
            l2 - Document sentences
    returns a list of document sentence indexes for each sentence in the summary 
    '''
    l = l1+l2
    sents_encodings = get_sen_encoding(l)
    similarities=cosine_similarity(sents_encodings)
    
    result = []
    for i in range(len(l1)):
        vals = similarities[i]
        vals = vals[len(l1):]
        idx = np.argmax(vals)
        result.append(idx)
    return result


def get_chunks_data_from_docV2(doc, summ):
    '''
    Function to generate chunks along with their summaries 
    input:  doc - legal Document
            summ - Gold standard summary
    returns a list of chunks and their summaries 
    '''
    chunk_summ_word_threshold = 150
    sentence_mapping = {}
    doc_sents = split_to_sentences(doc)
    summ_sents = split_to_sentences(summ)
    
    result = (similarity_l_l(summ_sents,doc_sents))
    
    for i in range(len(summ_sents)):
        sentence_mapping[doc_sents[result[i]]] = summ_sents[i]
    
    final_chunks = []
    final_summ = []
    for chunk in nest_sentencesV2(doc, 1024):
        summ = ""
        for chunk_sent in chunk:
            if chunk_sent in sentence_mapping:
                summ = summ + sentence_mapping[chunk_sent]
        if len(tokenizer.tokenize(summ)) >= chunk_summ_word_threshold:
            final_chunks.append(" ".join(chunk))
            final_summ.append(summ)
    return final_chunks, final_summ



training_chunks = []
training_summs = []
for i in tqdm(range(len(data_source))):
    cks, summs = get_chunks_data_from_docV2(data_source[i],data_summary[i])
    training_chunks = training_chunks + cks
    training_summs = training_summs + summs
#     print(i, len(training_summs), end = ", ", sep = " : ")
    if i%100 == 0: 
        full = list(zip(training_chunks,training_summs))
        df = pd.DataFrame(full,columns=['data', 'summary'])
        df.to_excel("FD_"+dataset+"_CLS_BK.xlsx")
#         break
full = list(zip(training_chunks,training_summs))
df = pd.DataFrame(full,columns=['data', 'summary'])
df.to_excel("FD_"+dataset+"_CLS.xlsx")


