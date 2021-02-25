import bz2
import requests
import json
from fuzzywuzzy import fuzz
import os
import re
import sys
sys.path.append('./bertserini/')
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer
from bertserini.retriever.pyserini_retriever import retriever, build_searcher

wiki_map = json.load(open('/data/mo.169/CQD4QA/85/HybridQA/data/wiki.json', 'r'))

def get_hyper_linked_para(wiki_map, cand_facts_hop1):

    hyper_linked_paragraphs = []
    for i in range(10): # consider top10 paragraphs in hop1
        title = cand_facts_hop1[i][0]

        try:
            paragraph_with_hyper_links = wiki_map[title]['text_with_links']
        except KeyError:
            continue

        # extract hyper links
        links = re.findall(r'href=\"(.*?)\">', paragraph_with_hyper_links) # ['Beetlejuice', 'Edward%20Scissorhands']

        for link in links:
            decoded_link = requests.utils.unquote(link)
            try:
                cand_para = wiki_map[decoded_link]
                if [decoded_link,cand_para['text']] not in hyper_linked_paragraphs:
                    hyper_linked_paragraphs.append([decoded_link,cand_para['text']])
            except KeyError:
                continue
    #print(len(hyper_linked_paragraphs))
    return hyper_linked_paragraphs

def retrieve_hyper_linked_para(title):

    hyper_linked_paragraphs = []

    try:
        paragraph_with_hyper_links = wiki_map[title]['text_with_links']
        # extract hyper links
        links = re.findall(r'href=\"(.*?)\">', paragraph_with_hyper_links) # ['Beetlejuice', 'Edward%20Scissorhands']
    except KeyError:
        links = []

    for link in links:
        decoded_link = requests.utils.unquote(link)
        try:
            cand_para = wiki_map[decoded_link]
            if [decoded_link,cand_para['text']] not in hyper_linked_paragraphs:
                hyper_linked_paragraphs.append([decoded_link,cand_para['text']])
        except KeyError:
            continue
    #print(len(hyper_linked_paragraphs))
    return hyper_linked_paragraphs

def load_data(path):
    # load hotpot qa dataset
    data = json.load(open(path))

    return data

def retrieve_cand_paragraph_wiki(question, paragraph, first_hop_title, searcher, wiki_map):

    hyper_linked_paragraphs = retrieve_hyper_linked_para(wiki_map, first_hop_title)
    #print(len(hyper_linked_paragraphs))
    cand_facts_hop2_ = retrieve_cand_paragraph(question, paragraph, searcher)

    if len(hyper_linked_paragraphs) >= 100:
        cand_facts_hop2 = hyper_linked_paragraphs[0:100]
    else:
        cand_facts_hop2 = hyper_linked_paragraphs + cand_facts_hop2_[:(100-len(hyper_linked_paragraphs))]
    
    return cand_facts_hop2

def retrieve_cand_paragraph_wiki_only_hyper(first_hop_title, wiki_map):

    hyper_linked_paragraphs = retrieve_hyper_linked_para(first_hop_title)

    return hyper_linked_paragraphs

def retrieve_cand_paragraph(question, paragraph, searcher):
    query = question if paragraph == '' else paragraph # question + ' ' + paragraph
    q = Question(query)
    # fetch some contexts from Wikipedia with Pyserini
    contexts = retriever(q, searcher, 100)

    candidate_facts = []
    for context in contexts:
        candidate_facts.append(context.text.split(' . '))

    return candidate_facts

def evaluate(path): 

    with open(path, 'r') as f:
        data = f.readlines()

    total_score = 0
    total_score_hop1 = 0
    total_score_hop2 = 0
    for i, item in enumerate(data[0:500]):
        #print(i)
        item = eval(item)
        supp_hop1 = item['supporting_facts_hop1']
        supp_hop2 = item['supporting_facts_hop2']
        cand_hop1 = item['candidate_facts_hop1']
        cand_hop2 = item['candidate_facts_hop2']

        cands1 = [item[1] for item in cand_hop1]
        for cand in cands1:
            score = fuzz.ratio(supp_hop1[1], cand)
            if score >= 80:
                total_score += 1
                total_score_hop1 += 1
                break
        # if supp_hop1[1] in cands1:
        #     total_score += 1
        #     total_score_hop1 += 1

        cands2 = [item[1] for item in cand_hop2]
        for cand in cands2:
            score = fuzz.ratio(supp_hop2[1], cand)
            if score >= 80:
                total_score += 1
                total_score_hop2 += 1 
                break       
        # if supp_hop2[1] in cands2:
        #     total_score += 1
        #     total_score_hop2 += 1

    print('Total: ', total_score, total_score/(2*len(data[0:500])))
    print('Hop1: ', total_score_hop1, total_score_hop1/len(data[0:500])) # 0.746
    print('Hop2: ', total_score_hop2, total_score_hop2/len(data[0:500])) # 0.772

def fast_construct(): # add hyper-linked paragraphs into candidates, save a new file named cand_paragraphs_hyperlink.txt
    cand_paragraphs = open('/home/mo.169/Projects/CQD4QA/text_retrieval/bertserini/output/train_data/cand_paragraphs.txt').readlines()
    save_path = '/home/mo.169/Projects/CQD4QA/text_retrieval/bertserini/output/train_data/cand_paragraphs_hyperlink.txt'

    with open(save_path, 'w') as f:
        for i, item in enumerate(cand_paragraphs):
            print(i)
            item = eval(item)
            new_item = {}
            new_item['id'] = item['id']
            new_item['question'] = item['question']
            new_item['answer'] = item['answer']
            new_item['supporting_facts_hop1'] = item['supporting_facts_hop1']
            new_item['supporting_facts_hop2'] = item['supporting_facts_hop2']
            new_item['candidate_facts_hop1'] = item['candidate_facts_hop1']  

            try:
                hyper_linked_paragraphs = get_hyper_linked_para(wiki_map, item['candidate_facts_hop1'] )
                cand_facts_hop2_ = item['candidate_facts_hop2']
                if len(hyper_linked_paragraphs) >= 100:
                    cand_facts_hop2 = hyper_linked_paragraphs[0:100]
                else:
                    cand_facts_hop2 = hyper_linked_paragraphs + cand_facts_hop2_[:(100-len(hyper_linked_paragraphs))]

                new_item['candidate_facts_hop2'] = cand_facts_hop2 
            except IndexError:
                new_item['candidate_facts_hop2'] = item['candidate_facts_hop2']

            f.write(str(new_item)+'\n')


def main():

    model_name = "rsvp-ai/bertserini-bert-base-squad"
    tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    searcher = build_searcher("/data/mo.169/index/lucene-index.enwiki-20180701-paragraphs")

    # hotpot data path
    dev_fullwiki_path = '/data/mo.169/HotpotQA/hotpot_dev_fullwiki_v1.json'
    dev_distractor_path = '/data/mo.169/HotpotQA/hotpot_dev_distractor_v1.json'
    train_path = '/data/mo.169/HotpotQA/hotpot_train_v1.1.json'

    hotpot_data = load_data(dev_distractor_path)

    outputs = []
    path = '/home/mo.169/Projects/CQD4QA/text_retrieval/bertserini/output/dev_data/cand_paragraphs_only_hyperlink.txt'

    with open(path, 'w') as f:
        for i, item in enumerate(hotpot_data):
            print(i)
            output = {}
            output['id'] = item['_id']
            question = item['question']
            output['question'] = question
            output['answer'] = item['answer']

            supporting_titles = []
            for x in item['supporting_facts']:
                if x[0] not in supporting_titles:
                    supporting_titles.append(x[0])

            for cont in item['context']:
                if cont[0] == supporting_titles[0]:
                    output['supporting_facts_hop1'] = [cont[0], ''.join(cont[1])]
                if cont[0] == supporting_titles[1]:
                    output['supporting_facts_hop2'] = [cont[0], ''.join(cont[1])]
            
            # the first hop
            try:
                cand_facts_hop1 = retrieve_cand_paragraph(question,'', searcher)
                hop1_top1_paragraph = cand_facts_hop1[0][1]
                hop1_top1_title = cand_facts_hop1[0][0]
                output['candidate_facts_hop1'] = cand_facts_hop1

                # the second hop
                hyper_linked_paragraphs = get_hyper_linked_para(wiki_map, cand_facts_hop1)
                #print(len(hyper_linked_paragraphs))
                cand_facts_hop2_ = retrieve_cand_paragraph(question, hop1_top1_paragraph, searcher)
                if len(hyper_linked_paragraphs) >= 100:
                    cand_facts_hop2 = hyper_linked_paragraphs[0:100]
                else:
                    cand_facts_hop2 = hyper_linked_paragraphs + cand_facts_hop2_[:(100-len(hyper_linked_paragraphs))]
                #print(len(cand_facts_hop2))
                output['candidate_facts_hop2'] = cand_facts_hop2
            except Exception as e:
                print('Error: ', e)
                output['candidate_facts_hop1'] = []               
                output['candidate_facts_hop2'] = []

            f.write(str(output)+'\n')
            outputs.append(output)

    evaluate(path)

if __name__ == '__main__':
    #create_wiki_map()
    main()
    # evaluate('/home/mo.169/Projects/CQD4QA/text_retrieval/bertserini/output/dev_data/cand_paragraphs_hyperlink.txt')
    # evaluate('/home/mo.169/Projects/CQD4QA/text_retrieval/bertserini/output/dev_data/cand_paragraphs.txt')
    # fast_construct()