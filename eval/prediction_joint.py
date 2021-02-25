import os
import sys
import pdb
import argparse
import numpy as np
from numpy import nan
from tqdm import tqdm
import random
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from ranker_joint import Ranker
import copy
import sys
sys.path.append('../reader/')
from reader import reader_predict
sys.path.append('../freebase/')
from freebase import Freebase

np.random.seed(123)

# freebase_url = "http://164.107.116.56:8890/sparql"
# freebase = Freebase(freebase_url)

ranker_path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_1.pt'

dev_final_ans = [ eval(x) for x in open('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/output/dev_data_merge/final_answers.txt').readlines() ]

example_context = [{"paragraphs": [{"qas": [{"question": "Were Scott Derrickson and Ed Wood of the same nationality?", \
    "is_impossible": False, \
        "answers": [{"text": "yes", "answer_start": -1}], \
        "id": "5a8b57f25542995d1e6f1371"}], \
            "context": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\" Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."}], \
                "title": "Doctor Strange (2016 film)"}]

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]

def merge_cwq_inference_data(args):
    # CWQ
    with open(args.cwq_kb_path, 'r') as f1:
        cwq_kb = f1.readlines()

    with open(args.cwq_text_path, 'r') as f2:
        cwq_text = f2.readlines()

    with open(args.cwq_merge_path, 'w') as f3:
        for idx, (kb, text) in enumerate(zip(cwq_kb[0:args.kb_eval_size], cwq_text[0:args.kb_eval_size])):
            merge_data = {}
            kb = eval(kb)
            text = eval(text)
            merge_data['question'] = kb['question']
            merge_data['answer'] = dev_final_ans[idx]

            cand_reason_paths = []

            cand_reason_paths_score_1 = []
            for i in range(len(kb['hop1_cand'][0:args.kb_beam_size])):
                path_score = [ [ kb['hop1_cand'][i] + ' ' + path, ans, kb['hop1_score'][i]*score ] for path, score, ans in zip(kb['hop2_cand'][i], kb['hop2_score'][i], kb['hop2_ans'][i]) ]
                cand_reason_paths_score_1.extend(path_score)

            cand_reason_paths_score_1_sort = sorted(cand_reason_paths_score_1, key=lambda x:x[2], reverse=True)
            cand_reason_paths = [ x[0:2] for x in  cand_reason_paths_score_1_sort][0:args.kb_beam_size] 

            cand_reason_paths_score_2 = []
            for i in range(len(text['hop1_cand'][0:args.text_beam_size])): # set Beam
                path_score = [ [text['hop1_cand'][i] + ' ' + path, text['hop1_score'][i] * score] for path, score in zip(text['hop2_cand'][i], text['hop2_score'][i]) ]
                cand_reason_paths_score_2.extend(path_score)
            
            cand_reason_paths_score_2_sort = sorted(cand_reason_paths_score_2, key=lambda x:x[1], reverse=True)
            cand_reason_paths.extend( [ x[0:1] for x in  cand_reason_paths_score_2_sort][0:args.text_beam_size] )         
            
            merge_data['cand_reason_paths'] = cand_reason_paths
            f3.write(str(merge_data)+'\n')

def merge_htqa_inference_data(args):
    # HotpotQA

    with open(args.htqa_kb_path, 'r') as f1:
        htqa_kb = f1.readlines()

    with open(args.htqa_text_path, 'r') as f2:
        htqa_text = f2.readlines()

    with open(args.htqa_merge_path, 'w') as f3:
        for i, (kb, text) in enumerate(zip(htqa_kb[0:args.text_eval_size], htqa_text[0:args.text_eval_size])):
            merge_data = {}

            kb = eval(kb)
            text = eval(text)

            merge_data['question'] = text['question']
            merge_data['answer'] = text['answer']

            cand_reason_paths = []

            cand_reason_paths_score_1 = []
            for i in range(len(kb['hop1_cand'][0:args.kb_beam_size])):
                path_score = [ [ kb['hop1_cand'][i] + ' ' + path, ans, kb['hop1_score'][i]*score ] for path, score, ans in zip(kb['hop2_cand'][i], kb['hop2_score'][i], kb['hop2_ans'][i]) ]
                cand_reason_paths_score_1.extend(path_score)

            cand_reason_paths_score_1_sort = sorted(cand_reason_paths_score_1, key=lambda x:x[2], reverse=True)
            cand_reason_paths = [ x[0:2] for x in  cand_reason_paths_score_1_sort][0:args.kb_beam_size] 

            cand_reason_paths_score_2 = []
            for i in range(len(text['hop1_cand'][0:args.text_beam_size])): # set Beam
                path_score = [ [text['hop1_cand'][i] + ' ' + path, text['hop1_score'][i] * score] for path, score in zip(text['hop2_cand'][i], text['hop2_score'][i]) ]
                cand_reason_paths_score_2.extend(path_score)
            
            cand_reason_paths_score_2_sort = sorted(cand_reason_paths_score_2, key=lambda x:x[1], reverse=True)
            cand_reason_paths.extend( [ x[0:1] for x in  cand_reason_paths_score_2_sort][0:args.text_beam_size] )   

            merge_data['cand_reason_paths'] = cand_reason_paths
            f3.write(str(merge_data)+'\n')


def predict_ans(args, ranker, tokenizer, dataset):
    torch.set_grad_enabled(False)
    ranker.train(False)
    device = torch.device("cuda:0")

    pred_answers = []
    gold_answers = []
    pred_reason_path = []
    if args.dataset_name == 'cwq':
        save_path = args.cwq_ans_path
    elif args.dataset_name == 'htqa':
        save_path = args.htqa_ans_path
    with open(save_path, 'w') as f:
        for i, data in enumerate(dataset):
            print(i)
            try:
                data = eval(data)
                question = data['question']
                question = tokenizer.batch_encode_plus(
                    [question], add_special_tokens=True, pad_to_max_length=True, return_tensors='pt')

                cand_reason_paths = data['cand_reason_paths']
                cand_reason_paths_ = [ (data['question'], x[0]) for x in cand_reason_paths ]

                cand_reason_paths_ids = tokenizer.batch_encode_plus(
                    cand_reason_paths_, add_special_tokens=True, padding=True, return_tensors='pt', truncation=True)

                outputs_1 = ranker(
                    cand_path_input_ids=cand_reason_paths_ids['input_ids'].to(device),
                    cand_path_token_type_ids=cand_reason_paths_ids['token_type_ids'].to(device),
                    cand_path_attention_mask=cand_reason_paths_ids['attention_mask'].to(device),
                    reason_paths_labels=None,
                    batch_size=1,
                    return_dict=True            
                )

                cand_score = outputs_1['cand_scores']
                # save candidate reasoning paths after ranker
                pred_reason_path.append({'cand_reason_paths':cand_reason_paths, 'cand_scores':cand_score[0].cpu().tolist()})

                # max_index = torch.argmax(cand_score, dim=1)[0]
                max_score = torch.max(cand_score, dim=1).values[0]
                max_index = 0
                for i, x in enumerate(cand_score[0]):
                    if x == max_score:
                        max_index = i
                        break
                if len(cand_reason_paths[max_index]) == 2:
                    answer = cand_reason_paths[max_index][1]
                    # pred_answers.append(answer)
                
                elif len(cand_reason_paths[max_index]) == 1:

                    new_context = copy.deepcopy(example_context)

                    new_context[0]['paragraphs'][0]['qas'][0]['question'] = data['question']
                    new_context[0]['paragraphs'][0]['context'] = cand_reason_paths[max_index][0]
                    
                    prediction = reader_predict(new_context)
                    answer = list(prediction.values())[0]

                f.write(str([answer, data['answer'], cand_reason_paths[max_index][0] ]) + '\n')
            except Exception as e:
                print('Error: ', e)
                if args.dataset_name == 'cwq':
                    f.write(str([[''], data['answer']]) + '\n')
                elif args.dataset_name == 'htqa':
                    f.write(str(['', data['answer']]) + '\n')


    return pred_answers, gold_answers

def eval_hybrid_cwq(pred_ans, gold_ans, wiki_title_to_mid):

    all_f1 = []
    all_precison = []
    all_recall = []
    trans_pred_ans = []
    for item in pred_ans:
        if isinstance(item, str):
            mid = wiki_title_to_mid(item)
            # print( 'wiki_title_to_mid: ', str((item, mid)) )
            trans_pred_ans.append([mid])
        else:
            trans_pred_ans.append(item)

    for i, (pred, gold) in enumerate(zip(trans_pred_ans, gold_ans)):
        # print(i)
        pred = list(set(pred))
        gold = list(set(gold))
        f1, precison, recall = f1_score_list(pred, gold)

        all_f1.append(f1)
        all_precison.append(precison)
        all_recall.append(recall)
    
    hit, hit_list = kb_hit_1(trans_pred_ans, gold_ans)

    return sum(all_f1)/len(all_f1), sum(all_precison)/len(all_precison), sum(all_recall)/len(all_recall), hit, hit_list, all_f1

def eval_hybrid_htqa(pred_ans, gold_ans, m2n_mappings):
    all_f1 = []
    all_em = []
    trans_pred_ans = []
    for item in pred_ans:
        if isinstance(item, list):
            entity_names = []
            for mid in item:
                try:
                    # name = freebase.find_name(mid)[0]
                    name = m2n_mappings[mid]
                except:
                    name = mid
                entity_names.append(name)
            print('m2n: ', str(item), str(entity_names))
            entity_seq = ' '.join(entity_names)
            trans_pred_ans.append(entity_seq)
        else:
            trans_pred_ans.append(item)

    for pred, gold in zip(trans_pred_ans, gold_ans):
        f1, _, _ = f1_score(pred, gold)
        em = exact_match_score(pred, gold)
        all_f1.append(f1)
        all_em.append(em)

    return sum(all_f1)/len(all_f1), all_f1, sum(all_em)/len(all_em), all_em

def eval_oracle_hybrid_htqa():
    # Htqa F1 = max( F1(gold, KB_pred),  F1(gold, Text_pred) )
    text_ans_data = open('/data/mo.169/CQD4QA/inference/htqa_ans/answers_7405_kb_0_text_5_new_cands_save_5.txt').readlines()
    kb_ans_data = open('/data/mo.169/CQD4QA/inference/htqa_ans/answers_7405_kb_5_text_0.txt').readlines()

    text_pred_answers = []
    gold_answers = []
    for item in text_ans_data:
        item = eval(item)
        text_pred_answers.append(item[0])
        gold_answers.append(item[1])

    kb_pred_answers = []
    for item in kb_ans_data:
        item = eval(item)
        kb_pred_answers.append(item[0])  

    m2n_mappings = json.load(open('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/data/m2n_cache.json'))


    text_f1, text_f1_list, text_em, text_em_list = eval_hybrid_htqa(text_pred_answers, gold_answers, m2n_mappings)
    kb_f1, kb_f1_list, kb_em, kb_em_list = eval_hybrid_htqa(kb_pred_answers, gold_answers, m2n_mappings)

    print('text_f1: ', text_f1)
    print('kb_f1: ', kb_f1)
    final_f1_list = []
    for text_f1, kb_f1 in zip(text_f1_list, kb_f1_list):
        final_f1_list.append(max(text_f1, kb_f1))
    
    final_f1 = sum(final_f1_list) / len(final_f1_list)
    print('final_f1: ', final_f1)

    print('text_em: ', text_em)
    print('kb_em: ', kb_em)
    final_em_list = []
    for text_em, kb_em in zip(text_em_list, kb_em_list):
        final_em_list.append(max(text_em, kb_em))
    
    final_em = sum(final_em_list) / len(final_em_list)
    print('final_em: ', final_em)

def eval_oracle_hybrid_cwq():
    # F1 = max( F1(gold, KB_pred),  F1(gold, Text_pred) )
    text_ans_data = open('/data/mo.169/CQD4QA/inference/cwq_ans/answers_3519_kb_0_text_5.txt').readlines()
    kb_ans_data = open('/data/mo.169/CQD4QA/inference/cwq_ans/answers_3519_kb_5_text_0.txt').readlines()

    text_pred_answers = []
    gold_answers = []
    for item in text_ans_data:
        item = eval(item)
        text_pred_answers.append(item[0])
        gold_answers.append(item[1])

    kb_pred_answers = []
    for item in kb_ans_data:
        item = eval(item)
        kb_pred_answers.append(item[0])  

    sys.path.append('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/')
    from map_title_mid import wiki_title_to_mid

    print('wiki_title_to_mid imported!')

    text_f1,_,_,text_hit, text_hit_list, text_f1_list = eval_hybrid_cwq(text_pred_answers, gold_answers, wiki_title_to_mid)
    print('text_f1: ', text_f1)
    print('text_hit: ', text_hit)
    kb_f1,_,_,kb_hit, kb_hit_list, kb_f1_list = eval_hybrid_cwq(kb_pred_answers, gold_answers, wiki_title_to_mid)
    print('kb_f1: ', kb_f1)
    print('kb_hit: ', kb_hit)

    final_f1_list = []
    for text_f1, kb_f1 in zip(text_f1_list, kb_f1_list):
        final_f1_list.append(max(text_f1, kb_f1))
    final_f1 = sum(final_f1_list) / len(final_f1_list)
    print('final_f1: ', final_f1)

    final_hit_list = []
    for text_hit, kb_hit in zip(text_hit_list, kb_hit_list):
        final_hit_list.append(max(text_hit, kb_hit))
    final_hit = sum(final_hit_list) / len(final_hit_list)
    print('final_hit: ', final_hit)

def kb_hit_1(pred_ans, gold_ans):
    hit = []

    for pred, gold in zip(pred_ans, gold_ans):
        mark = False
        try:
            one_ans = random.sample(pred, 1)[0]
        except ValueError:
            one_ans = ''
        if one_ans in gold:
            mark = True
        if mark:
            hit.append(1)
        else:
            hit.append(0)
    
    return sum(hit)/len(hit), hit

def test_reader_oracle():
    htqa = json.load(open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_squad_v2.0_format.json'))
    pred_answers = []
    gold_answers = []

    for item in htqa['data']:
        gold_answer = item['paragraphs'][0]['qas'][0]['answers'][0]['text']
        gold_answers.append(gold_answer)
        # prediction = reader_predict([item])
        # pred_answer = list(prediction.values())[0]
        # pred_answers.append(pred_answer)
        # print(str((pred_answer, gold_answer)))

        new_context = copy.deepcopy(example_context)

        new_context[0]['paragraphs'][0]['qas'][0]['question'] = item['paragraphs'][0]['qas'][0]['question']
        new_context[0]['paragraphs'][0]['context'] = item['paragraphs'][0]['context']
        prediction = reader_predict(new_context)
        pred_answer = list(prediction.values())[0]
        pred_answers.append(pred_answer)
        print(str((pred_answer, gold_answer)))
    
    f1, _, em, _ = eval_hybrid_htqa(pred_answers, gold_answers, {})
    print('F1: ', f1)
    print('EM: ', em)


def main():

    kb_eval_size = 3519 # 3519
    text_eval_size = 7405 # 7405
    kb_beam_size = 5
    text_beam_size = 5

    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--dataset_name', type=str, default='htqa', help='cwq or htqa')
    parser.add_argument('--kb_eval_size', type=int, default=kb_eval_size) # 3519
    parser.add_argument('--text_eval_size', type=int, default=text_eval_size) # 7405
    parser.add_argument('--kb_beam_size', type=int, default=kb_beam_size)
    parser.add_argument('--text_beam_size', type=int, default=text_beam_size)

    parser.add_argument('--cwq_kb_path', type=str, default='/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/cwq_kb.txt')
    parser.add_argument('--cwq_text_path', type=str, default='/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/cwq_text.txt')
    parser.add_argument('--cwq_merge_path', type=str, default='/data/mo.169/CQD4QA/inference/cwq_merge/inference_%s_kb_%s_text_%s.txt'%(kb_eval_size, kb_beam_size, text_beam_size))
    parser.add_argument('--cwq_ans_path', type=str, default='/data/mo.169/CQD4QA/inference/cwq_ans/answers_%s_kb_%s_text_%s.txt'%(kb_eval_size, kb_beam_size, text_beam_size))

    parser.add_argument('--htqa_text_path', type=str, default='/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/htqa_text.txt')
    parser.add_argument('--htqa_kb_path', type=str, default='/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/htqa_kb.txt')
    parser.add_argument('--htqa_merge_path', type=str, default='/data/mo.169/CQD4QA/inference/htqa_merge/inference_%s_kb_%s_text_%s.txt'%(text_eval_size, kb_beam_size, text_beam_size))
    parser.add_argument('--htqa_ans_path', type=str, default='/data/mo.169/CQD4QA/inference/htqa_ans/answers_%s_kb_%s_text_%s.txt'%(text_eval_size, kb_beam_size, text_beam_size))

    args = parser.parse_args()

    # merge inference data and load
    if args.dataset_name == 'cwq':
        merge_cwq_inference_data(args)
        with open(args.cwq_merge_path, 'r') as f:
            data = f.readlines()

    elif args.dataset_name == 'htqa':
        merge_htqa_inference_data(args)
        m2n_mappings = json.load(open('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/data/m2n_cache.json'))
        with open(args.htqa_merge_path, 'r') as f:
            data = f.readlines()

    # Build model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda")
    args.device = device
    ranker = Ranker(args, hid=768).to(device)

    # load checkpoints path
    ranker.load_state_dict(torch.load(ranker_path))

    # predict answer
    pred_answers, gold_answers = predict_ans(args, ranker, tokenizer, data)

    # load answer
    if args.dataset_name == 'cwq':
        with open(args.cwq_ans_path, 'r') as f:
            ans_data = f.readlines()
    elif args.dataset_name == 'htqa':
        with open(args.htqa_ans_path, 'r') as f:
            ans_data = f.readlines()

    pred_answers = []
    gold_answers = []
    for item in ans_data:
        item = eval(item)
        pred_answers.append(item[0])
        gold_answers.append(item[1])
    
    # evaluate
    if args.dataset_name == 'cwq':
        sys.path.append('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/')
        from map_title_mid import wiki_title_to_mid
        f1, precision, recall,hit,_, _ = eval_hybrid_cwq(pred_answers, gold_answers, wiki_title_to_mid)
        print('F1: ', f1)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('Hit@1: ', hit)
    elif args.dataset_name == 'htqa':
        f1, _, em, _ = eval_hybrid_htqa(pred_answers, gold_answers, m2n_mappings)
        print('F1: ', f1)
        print('EM: ', em)

if __name__ == '__main__':
    main()
    # eval_oracle_hybrid_htqa()
    # eval_oracle_hybrid_cwq()
    # test_reader_oracle()