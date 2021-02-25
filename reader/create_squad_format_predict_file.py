'''
Convert our predicted file into squad format file
'''
import numpy as np
import json
import sys
sys.path.append('/home/mo.169/Projects/CQD4QA/')
from eval_qa import *
import copy

def prepare_top1():

    # pred_path = '/data/mo.169/CQD4QA/inference/only_htqa_ans/reason_path_bm_40_hy_10_15999_new.txt'
    pred_path = '/data/mo.169/CQD4QA/inference/only_htqa_ans/reason_path_bm_40_hy_10_epoch_2_ranker_epoch_2_beam_10.txt'
    
    pred_file = open(pred_path, 'r').readlines()

    example_data = json.load(open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_squad_v2.0_format.json'))

    example_file = example_data.copy()
    example_file['data'] = example_file['data'][0:len(pred_file)]
    example_file_json = json.dumps(example_file)
    with open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_%s_squad_v2.0_format.json'%(str(len(pred_file))), 'w') as f:
        f.write(example_file_json)

    our_file = example_file.copy()

    for i in range(len(pred_file)):
        pred_item = eval(pred_file[i].strip())
        max_id = np.argmax(pred_item['cand_scores'])
        prediction = pred_item['cand_reason_paths'][max_id][0]
        our_file['data'][i]['paragraphs'][0]['context'] = prediction

    our_file['data'] = our_file['data'][0:len(pred_file)]

    our_file_json = json.dumps(our_file)
    with open('./eval_data/pred_%s_squad_v2.0_format.json'%(str(len(pred_file))), 'w') as f:
        f.write(our_file_json)

def prepare_topk():

    pred_path = '/data/mo.169/CQD4QA/inference/only_htqa_merge/extend_inference_bm_20_hy_10_epoch_2_new_ranker_beam_8.txt'
    
    pred_file = open(pred_path, 'r').readlines()

    gold_file = json.load(open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_squad_v2.0_format.json'))

    our_file = {'version':'v2.0', 'data':[]}

    paragraphs = []
    for i, (pred, gold) in enumerate(zip(pred_file[:], gold_file['data'][:])):
        pred_item = eval(pred.strip())

        for j in pred_item['cand_reason_paths']:
            cell = copy.deepcopy(gold) # deep copy
            cell['paragraphs'][0]['context'] = j[0]
            our_file['data'].append(cell)

    our_file_json = json.dumps(our_file)
    with open('./eval_data/pred_bm_20_hy_10_epoch_2_beam_8.json', 'w') as f:
        f.write(our_file_json)

def oracle_test():

    example_file = json.load(open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_squad_v2.0_format.json'))

    our_file = example_file.copy()

    htpt = json.load(open('/data/mo.169/HotpotQA/hotpot_dev_distractor_v1.json', 'r'))

    for i in range(500):
        item = htpt[i]
        supporting_titles = []
        for x in item['supporting_facts']:
            if x[0] not in supporting_titles:
                supporting_titles.append(x[0])

        gold_list = []
        for cont in item['context']:
            if cont[0] == supporting_titles[0]:
                gold_list.append( ''.join(cont[1]) )
            if cont[0] == supporting_titles[1]:
                gold_list.append( ''.join(cont[1]) )
        gold = ' '.join(gold_list)

        our_file['data'][i]['paragraphs'][0]['context'] = gold

    our_file['data'] = our_file['data'][0:500]

    our_file_json = json.dumps(our_file)
    with open('./eval_data/gold_squad_v2.0_format.json', 'w') as f:
        f.write(our_file_json)

def eval_qa(pred_ans, gold_ans):

    all_f1 = []
    for pred, gold in zip(pred_ans, gold_ans):
        
        f1,_,_ = f1_score(pred, gold)
        all_f1.append(f1)

    return sum(all_f1)/(len(all_f1))

def eval_htpt_way():
    pred_file = json.load(open('/data/mo.169/ICLR2020/models/reader/predictions.json'))
    # gold_file = json.load(open('/data/mo.169/ICLR2020/data/hotpot/hotpot_dev_500_squad_v2.0_format.json'))
    gold_file = json.load(open('/data/mo.169/HotpotQA/hotpot_dev_distractor_v1.json', 'r'))

    pred_answers = []
    gold_answers = []
    for i in range(len(list( pred_file.keys() ))):
        ids = list( pred_file.keys() )
        idx = ids[i]
        pred_ans = pred_file[idx]
        # print('pred_ans: ', pred_ans)
        pred_answers.append(pred_ans)
        gold_ans = gold_file[i]['answer']
        # print('gold_ans: ', gold_ans)
        gold_answers.append(gold_ans)
        assert idx == gold_file[i]['_id']

    f1 = eval_qa(pred_answers, gold_answers)
    print(f1)

if __name__ == '__main__':
    # main()
    # oracle_test()
    # eval_htpt_way()
    prepare_top1()
    # prepare_topk()