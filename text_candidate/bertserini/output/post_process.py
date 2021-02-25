'''
(1)Fuzzy matching
(2)Insert golden paragraph into the 100 candidate paragraphs
'''
from fuzzywuzzy import fuzz


path = './dev_data/original/cand_paragraphs.txt'

with open(path, 'r') as f:
    data = f.readlines()
'''
path_1 = './train_data/cand_paragraphs_part1.txt'
with open(path_1, 'r') as f:
    data1 = f.readlines()

path_2 = './train_data/cand_paragraphs_part2.txt'
with open(path_2, 'r') as f:
    data2 = f.readlines()

data = data1 + data2
'''
#print(len(data))
with open('./dev_data/cand_paragraphs_.txt', 'w') as f:
    for i, item in enumerate(data[0:10]):
        print(i)
        item = eval(item)
        output = {}
        output['id'] = item['id']
        output['question'] = item['question']
        output['answer'] = item['answer']
        output['supporting_facts_hop1'] = item['supporting_facts_hop1']
        output['supporting_facts_hop2'] = item['supporting_facts_hop2']

        try:
            max_score_1 = 0
            max_index_1 = 0
            for index, fact_hop1 in enumerate(item['candidate_facts_hop1']):
                #print(item['supporting_facts_hop1'][1])
                #print(fact_hop1[1])
                
                score = fuzz.ratio(item['supporting_facts_hop1'][1], fact_hop1[1])     
                if score > max_score_1:
                    max_score_1 = score
                    max_index_1 = index
            #print('gold hop1: ', item['supporting_facts_hop1'])
            #print('cand hop1: ', item['candidate_facts_hop1'][max_index_1])
            print('max_score_1', max_score_1)
            cand_facts_hop1_copy = item['candidate_facts_hop1'].copy()
            #print('cand_facts_hop1_copy:', cand_facts_hop1_copy)
            if max_score_1 >= 80:
                #print('Replace!')
                cand_facts_hop1_copy[max_index_1] = item['supporting_facts_hop1']
            output['candidate_facts_hop1'] = cand_facts_hop1_copy
        except IndexError:
            print('Error!')
            output['candidate_facts_hop1'] = []

        try:
            max_score_2 = 0
            max_index_2 = 0
            for index, fact_hop2 in enumerate(item['candidate_facts_hop2']):
                score = fuzz.ratio(item['supporting_facts_hop2'][1], fact_hop2[1])     
                if score > max_score_2:
                    max_score_2 = score
                    max_index_2 = index
            #print('gold hop2: ', item['supporting_facts_hop2'])
            #print('cand hop2: ', item['candidate_facts_hop2'][max_index_2])
            print('max_score_2', max_score_2)
            cand_facts_hop2_copy = item['candidate_facts_hop2'].copy()
            if max_score_2 >= 80:
                cand_facts_hop2_copy[max_index_2] = item['supporting_facts_hop2']
            output['candidate_facts_hop2'] = cand_facts_hop2_copy
        except IndexError:
            print('Error!')
            output['candidate_facts_hop2'] = []

        f.write(str(output)+'\n')