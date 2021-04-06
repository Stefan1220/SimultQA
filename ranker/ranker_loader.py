import re
import pdb
import json
import random
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def handle_corner_case(line):
    """
    Example:
    Complex question 'composition # the film has a charcter named junior lamb # 8 # who plays %composition'
    should be splitted into ['composition', 'the film has a charcter named junior lamb # 8',
    '# who plays %composition']. Can not simply use # as delimiter.
    """
    replace = {}
    cnt = 0
    # print(line)
    while True:
        find = re.search(r' # \d+ ', line)
        if not find: break
        replacement = f"SPECIAL_CASE_{cnt}"
        replace[replacement] = line[find.start() + 1:find.end() - 1]
        line = line[:find.start()] + ' ' + replacement + ' ' + line[find.end():]
        cnt += 1
    splits = line.split(" # ")
    newsplits = [splits[0]]
    for i in range(1, len(splits)):
        s = splits[i]
        for x, y in replace.items():
            s = s.replace(x, y, 1)
        newsplits.append(s)
    if len(newsplits) != 3:
        pdb.set_trace()
    # print(newsplits)
    return newsplits


def add_yes_no(string):
    # Allow model to explicitly select yes/no from text (location front, avoid truncation)
    return " ".join(["yes", "no", string])


def find_span(x, y):
    x = tokenizer.tokenize(x)
    y = tokenizer.tokenize(y)
    start_index = -1
    end_index = -1

    for i in range(len(x)-len(y)+1):
        x_slice = x[i:i+len(y)]
        if x_slice == y:
            start_index = i
            end_index = i + len(y) - 1

    if start_index == -1 and end_index == -1:
        start_index = len(x) - 1
        end_index = 0

    return start_index, end_index


class CWQRankerDatset(Dataset):
    def __init__(self, kb_data_path, text_data_path,
                 train=False, weak_train=False, num_kb_samples=20, num_text_samples=20):

        self.train = train
        self.weak_train = weak_train
        self.num_kb_samples = num_kb_samples
        self.num_text_samples = num_text_samples

        # dict_keys(['gold_hop1', 'gold_hop2', 'cand_hop1', 'cand_hop2', 'answers'])
        self.kb_data = pickle.load(open(kb_data_path, 'rb'))

        # dict_keys(['id', 'question', 'candidate_facts_hop1', 'candidate_facts_hop2'])
        self.text_data = pickle.load(open(text_data_path, 'rb'))
        assert len(self.kb_data) == len(self.text_data)

        self.questions = [x['question'].strip() for x in self.text_data]
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.train:
            # get the paths
            hop1_kb_gold = self.kb_data[idx]['gold_hop1'][0]
            hop2_kb_gold = self.kb_data[idx]['gold_hop2'][0]
            gold_kb_path = "{0} {1}".format(hop1_kb_gold, hop2_kb_gold)

            ## Sample kb negative reasoning paths for ranker training
            cur_hop1_kb_cands = [x[0] for x in self.kb_data[idx]['cand_hop1']]
            cur_hop2_kb_cands = [x[0] for x in self.kb_data[idx]['cand_hop2']]

            num_kb_samples = self.num_kb_samples

            # collect negative samples for each hop, then combine
            if len(cur_hop1_kb_cands) >= num_kb_samples:
                hop1_kb_cands = random.sample(cur_hop1_kb_cands, k=num_kb_samples)
            else:
                hop1_kb_cands = cur_hop1_kb_cands
                while len(hop1_kb_cands) < num_kb_samples:
                    temp_hop1 = self.kb_data[random.randint(0, self.len-1)]['gold_hop1'][0]
                    if temp_hop1 != '':
                        hop1_kb_cands.append(temp_hop1)
                    
            if len(cur_hop2_kb_cands) >= num_kb_samples:
                hop2_kb_cands = random.sample(cur_hop2_kb_cands, k=num_kb_samples)
            else:
                hop2_kb_cands = cur_hop2_kb_cands
                while len(hop2_kb_cands) < num_kb_samples:
                    temp_hop2 = self.kb_data[random.randint(0, self.len-1)]['gold_hop2'][0]
                    if temp_hop2 != '':
                        hop2_kb_cands.append(temp_hop2)

            assert len(hop1_kb_cands) == num_kb_samples
            assert len(hop2_kb_cands) == num_kb_samples

            num_type1 = num_kb_samples // 3
            num_type2 = num_kb_samples // 3
            num_type3 = num_kb_samples - (num_type1 + num_type2)

            # gold 1Hop + random 2Hop
            type1_samples = []
            type1_labels = []
            while len(type1_samples) < num_type1:
                rand_hop2 = random.choice(hop2_kb_cands)
                if rand_hop2 != hop2_kb_gold:
                    type1_samples.append("{} {}".format(hop1_kb_gold, rand_hop2))
                    type1_labels.append(0)

            # random 1Hop + gold 2Hop
            type2_samples = []
            type2_labels = []
            while len(type2_samples) < num_type2:
                rand_hop1 = random.choice(hop1_kb_cands)
                if rand_hop1 != hop1_kb_gold:
                    type2_samples.append("{} {}".format(rand_hop1, hop2_kb_gold))
                    type2_labels.append(0)

            # random 1Hop + random 2Hop
            type3_samples = []
            type3_labels = []
            while len(type3_samples) < num_type3:
                rand_hop1 = random.choice(hop1_kb_cands)
                rand_hop2 = random.choice(hop2_kb_cands)
                if rand_hop1 != hop1_kb_gold and rand_hop2 != hop2_kb_gold:
                    type3_samples.append("{} {}".format(rand_hop1, rand_hop2))
                    type3_labels.append(0)

            kb_samples = type1_samples + type2_samples + type3_samples
            kb_labels = type1_labels + type2_labels + type3_labels
            assert len(kb_samples) == num_kb_samples

            ## Sample text negative reasoning paths for ranker training
            hop1_text_cands = [x for x in self.text_data[idx]['candidate_facts_hop1']]  # [title, paragraph, score_1, score_2]
            hop2_text_cands = [x for x in self.text_data[idx]['candidate_facts_hop2']]
            num_text_samples = self.num_text_samples
            
            def text_score():
                return

            # random 1Hop + random 2Hop
            text_samples = []
            text_labels = []
            while len(text_samples) < num_text_samples:
                rand_hop1 = random.choice(hop1_text_cands)
                rand_hop2 = random.choice(hop1_text_cands)
                text_samples.append("{} {}".format(rand_hop2[1], rand_hop2[1]))
                if not self.weak_train:
                    text_labels.append(0)
                else:
                    if 
                    text_labels.append(0)

            
            assert len(text_samples) == num_text_samples

            all_samples = [gold_kb_path] + kb_samples + text_samples
            all_labels = np.array([1.] + kb_labels + text_labels)
            all_labels = all_labels / sum(all_labels)  # normalize ranking scores
            shuffle_idx = list(range(len(all_samples))); np.random.shuffle(shuffle_idx)
            all_samples = [all_samples[idx] for idx in shuffle_idx]
            all_labels = all_labels[shuffle_idx]
            # all_labels = [all_labels[idx] for idx in shuffle_idx]

            return {
                'question': self.questions[idx],
                'cand_reason_paths': all_samples,
                'reason_path_label': all_labels
            }
        else:
            num_neg_sampls = self.num_samples - 1

            hop1_gold = self.data[idx]['hop1_gold'][0]
            hop1_gold_aws = self.data[idx]['hop1_gold'][1]

            hop2_gold = self.data[idx]['hop2_gold'][0]
            hop2_gold_aws = self.data[idx]['hop2_gold'][1]

            hop1_cands = [x[0] for x in self.data[idx]['hop1_cands']]
            hop1_cand_aws = [x[1] for x in self.data[idx]['hop1_cands']]

            hop2_cands = [x[0] for x in self.data[idx]['hop2_cands']]
            hop2_cand_aws = [x[1] for x in self.data[idx]['hop2_cands']]

            cands = [hop1_cands, hop2_cands]
            cand_aws = [hop1_cand_aws, hop2_cand_aws]
            gold_paths = [hop1_gold, hop2_gold]
            gold_aws = [hop1_gold_aws, hop2_gold_aws]

            # sample negative samples as candidate reasoning path
            cand_reason_paths = []
            cand_reason_ans = []
            # answer = self.data[idx]['answer']

            hop1_cand_ans_pairs = self.data[idx]['hop1_cands']
            hop2_cand_ans_pairs = self.data[idx]['hop2_cands']
            
            if len(hop1_cand_ans_pairs) >= num_neg_sampls:
                sample_hop1_cands = random.sample(hop1_cand_ans_pairs, k=num_neg_sampls)
            else:
                sample_hop1_cands = hop1_cand_ans_pairs
                for i in range(num_neg_sampls-len(hop1_cand_ans_pairs) ):
                    try:
                        sample_hop1_cands.append(self.data[random.randint(0, self.len-1)]['hop1_cands'][0])
                    except:
                        print('Corner Case!')
                        sample_hop1_cands.append(['', ['']])

            if len(hop2_cand_ans_pairs) >= num_neg_sampls:
                sample_hop2_cands = random.sample(hop2_cand_ans_pairs, k=num_neg_sampls)
            else:
                sample_hop2_cands = hop2_cand_ans_pairs
                for i in range(num_neg_sampls-len(hop2_cand_ans_pairs) ):
                    try:
                        sample_hop2_cands.append(self.data[random.randint(0, self.len-1)]['hop2_cands'][0])      
                    except:
                        print('Corner Case!')
                        sample_hop2_cands.append(['', ['']])      

            count_part_1 = 0
            for i in range(num_neg_sampls):
                if count_part_1 >= int(num_neg_sampls/2):
                    break
                hop1 = sample_hop1_cands[i][0]
                if hop1 == hop1_gold:
                    continue
                else:
                    reason_path = str(hop1) + ' ' + str(hop2_gold)
                    cand_reason_paths.append(reason_path)
                    cand_reason_ans.append(hop2_gold_aws)
                    count_part_1 += 1

            count_part_2 = 0
            for i in range(num_neg_sampls):
                if count_part_2 >= (num_neg_sampls - int(num_neg_sampls/2)):
                    break
                hop2 = sample_hop2_cands[i][0]
                if hop2 == hop2_gold:
                    continue
                else:
                    reason_path = str(hop1_gold) + ' ' + str(hop2)
                    cand_reason_paths.append(reason_path)
                    cand_reason_ans.append(sample_hop2_cands[i][1])
                    count_part_2 += 1
            
            assert len(cand_reason_paths) == num_neg_sampls

            reason_path_label = np.random.randint(self.num_samples)
            cand_reason_paths.insert(reason_path_label, str(hop1_gold)+' '+str(hop2_gold))
            cand_reason_ans.insert(reason_path_label, hop2_gold_aws)

            return {
                'question': self.questions[idx],
                "decompositions": self.decompositions[idx],
                'decomp_cand_paths': cands,
                'decomp_gold_paths': gold_paths,
                'decomp_cand_aws': cand_aws,
                'decomp_gold_aws': gold_aws,
                'cand_reason_paths': cand_reason_paths,
                'reason_path_label': reason_path_label,
                'cand_reason_ans': cand_reason_ans
            }


class HotpotQARankerDataset(Dataset):
    def __init__(self, text_data_path, kb_data_path,
                 train=False, weak_labels=False, num_text_samples=20, num_kb_samples=20):

        self.train = train
        self.num_text_samples = num_text_samples
        self.num_kb_samples = num_kb_samples

        # dict_keys(['id', 'question', 'tfidf_candidates', 'hyperlink_candidates', 'supporting_facts_hop1', 'supporting_facts_hop2', 'answer'])        
        self.text_data = pickle.load(open(text_data_path, 'rb'))
        self.kb_data = pickle.load(open(kb_data_path, 'rb'))  # list
        assert len(self.kb_data) == len(self.text_data)

        self.questions = [x['question'].strip() for x in self.text_data]
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.train:
            hop1_title = self.text_data[idx]['supporting_facts_hop1'][0]
            hop2_title = self.text_data[idx]['supporting_facts_hop2'][0]
            hop1_gold = self.text_data[idx]['supporting_facts_hop1'][1]  # ['title', 'paragraph']
            hop2_gold = self.text_data[idx]['supporting_facts_hop2'][1]
            gold_answer = self.text_data[idx]['answer']

            all_tf_cand_paras = [x[1] for x in self.text_data[idx]['tfidf_candidates']
                if x[0] != hop1_title and x[0] != hop2_title and x[1] != None and x[1] != [] and x[1] != '']
            all_hy_cand_paras = [x[1] for x in self.text_data[idx]['hyperlink_candidates'] 
                if x[0] != hop1_title and x[0] != hop2_title and x[1] != None and x[1] != [] and x[1] != '']

            gold_text_path = "{} {}".format(hop1_gold, hop2_gold)
            num_text_samples = self.num_text_samples

            # Construct a pool of negative samples in each hop
            if len(all_tf_cand_paras) >= num_text_samples:
                tfidf_cands = all_tf_cand_paras[:num_text_samples]
            else:
                tfidf_cands = all_tf_cand_paras
                while len(tfidf_cands) < num_text_samples:
                    temp_hop1 = self.text_data[random.randint(0, self.len-1)]['supporting_facts_hop1']
                    if gold_answer.lower() not in temp_hop1[1].lower() and hop1_title != temp_hop1[0]:
                        tfidf_cands.append(temp_hop1)
                    
            if len(all_hy_cand_paras) >= num_text_samples:
                hyper_cands = all_hy_cand_paras[:num_text_samples]
            else:
                hyper_cands = all_hy_cand_paras
                while len(hyper_cands) < num_text_samples:
                    temp_hop2 = self.text_data[random.randint(0, self.len-1)]['supporting_facts_hop2']
                    if gold_answer.lower() not in temp_hop2[1].lower() and hop2_title != temp_hop2[0]:
                        hyper_cands.append(temp_hop2)

            assert len(tfidf_cands) == num_text_samples
            assert len(hyper_cands) == num_text_samples

            num_type1 = num_text_samples // 2
            num_type2 = num_text_samples - num_type1

            # sampled tfidf + gold hyperlink
            type1_samples = []
            type1_labels = []
            for neg_tfidf in tfidf_cands:
                type1_samples.append("{} {}".format(neg_tfidf, hop2_gold))
                type1_labels.append(0)
                if len(type1_samples) == num_type1:
                    break

            # gold tfidf + sampled hyperlink
            type2_samples = []
            type2_labels = []
            for neg_hyper in tfidf_cands:
                type2_samples.append("{} {}".format(hop1_gold, neg_hyper))
                type2_labels.append(0)
                if len(type2_samples) == num_type2:
                    break

            text_samples = type1_samples + type2_samples
            text_labels = type1_labels + type2_labels
            assert len(text_samples) == num_text_samples

            # Sample negative kb paths
            num_kb_samples = self.num_kb_samples
            kb_samples = []
            kb_labels = []

            all_kb_cand_paths = [x[0] for x in self.kb_data[idx]]
            if len(all_kb_cand_paths) >= num_kb_samples:
                kb_samples = random.sample(all_kb_cand_paths, k=num_kb_samples)
                kb_labels = [0] * len(kb_samples)
            else:
                kb_samples = all_kb_cand_paths
                kb_labels = [0] * len(kb_samples)
                while len(kb_samples) < num_kb_samples:
                    temp_data = self.kb_data[random.randint(0, self.len-1)]
                    if len(temp_data) > 0:
                        kb_samples.append(temp_data[random.randint(0, len(temp_data)-1)][0])
                        kb_labels.append(0)

            assert len(kb_samples) == num_kb_samples

            all_samples = [gold_text_path] + text_samples + kb_samples
            all_labels = np.array([1.] + text_labels + kb_labels)
            all_labels = all_labels / sum(all_labels)  # normalize ranking scores
            shuffle_idx = list(range(len(all_samples))); np.random.shuffle(shuffle_idx)
            all_samples = [all_samples[idx] for idx in shuffle_idx]
            all_labels = all_labels[shuffle_idx]
            # all_labels = [all_labels[idx] for idx in shuffle_idx]

            return {
                'question': self.questions[idx],
                'cand_reason_paths': all_samples,
                'reason_path_label': all_labels
            }

        else:
            num_neg_sampls = self.num_samples - 1
            hop1_gold = self.data[idx]['hop1_gold'][1]
            hop2_gold = self.data[idx]['hop2_gold'][1]

            hop1_cands = [x[1] for x in self.data[idx]['hop1_cands']]
            hop2_cands = [x[1] for x in self.data[idx]['hop2_cands']]

            cands = [hop1_cands, hop2_cands]
            gold_paths = [hop1_gold, hop2_gold]
            gold_reason_path = hop1_gold+' '+hop2_gold

            # sample negative samples as candidate reasoning path
            cand_reason_paths = []
            answer = self.data[idx]['answer']

            hop1_title_cand_pairs = self.data[idx]['hop1_cands']
            hop2_title_cand_pairs = self.data[idx]['hop2_cands']
            
            if len(hop1_title_cand_pairs) >= num_neg_sampls:
                sample_hop1_cands = random.sample(hop1_title_cand_pairs, k=num_neg_sampls)
            else:
                sample_hop1_cands = hop1_title_cand_pairs
                for i in range(num_neg_sampls-len(hop1_title_cand_pairs) ):
                    try:
                        sample_hop1_cands.append(self.data[random.randint(0, self.len-1)]['hop1_cands'][0])
                    except:
                        print('Corner Case!')
                        sample_hop1_cands.append(['', ''])

            if len(hop2_title_cand_pairs) >= num_neg_sampls:
                sample_hop2_cands = random.sample(hop2_title_cand_pairs, k=num_neg_sampls)
            else:
                sample_hop2_cands = hop2_title_cand_pairs
                for i in range(num_neg_sampls-len(hop2_title_cand_pairs) ):
                    try:
                        sample_hop2_cands.append(self.data[random.randint(0, self.len-1)]['hop2_cands'][0])      
                    except:
                        print('Corner Case!')
                        sample_hop2_cands.append(['', ''])     

            count_part_1 = 0
            for i in range(num_neg_sampls):
                if count_part_1 >= int(num_neg_sampls/2):
                    break
                hop1_title = sample_hop1_cands[i][0]
                if hop1_title == self.data[idx]['hop1_gold'][0]: # title same or not
                    continue
                else:
                    reason_path = sample_hop1_cands[i][1] + ' ' + hop2_gold
                    cand_reason_paths.append(reason_path)
                    count_part_1 += 1

            count_part_2 = 0
            for i in range(num_neg_sampls):
                if count_part_2 >= (num_neg_sampls - int(num_neg_sampls/2)):
                    break
                hop2_title = sample_hop2_cands[i][0]
                if hop2_title == self.data[idx]['hop2_gold'][0]:
                    continue
                else:
                    reason_path =  hop1_gold + ' ' + sample_hop2_cands[i][1]
                    cand_reason_paths.append(reason_path)
                    count_part_2 += 1

            reason_path_label = np.random.randint(self.num_samples)
            cand_reason_paths.insert(reason_path_label, gold_reason_path)

            # print('Dev cand reason paths lenght: ', len(cand_reason_paths))
            # prepare paragraph for reader
            reader_paragraph = str(hop1_gold)+' '+str(hop2_gold)
            reader_paragraph = add_yes_no(reader_paragraph) # add yes and no
            # prepare start and end position
            start_position, end_position = find_span(reader_paragraph, answer)

            return {
                'id': self.data[idx]['id'],
                'question': self.questions[idx],
                'answer': self.data[idx]['answer'],
                'decomp_cand_paths': cands,
                'decomp_gold_paths': gold_paths,
                'cand_reason_paths': cand_reason_paths,
                'reason_path_label': reason_path_label,
                'reader_paragraph': reader_paragraph,
                'start_position': start_position,
                'end_position': end_position
            }


if __name__ == '__main__':

    # train_set = CWQDatset('../data/demo_processed_kb_train.pkl', train=True, num_samples=20)
    # dev_set = CWQDatset('../data/demo_processed_kb_dev.pkl', train=False)

    # train_loader = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=0)
    # dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)

    # print(next(iter(train_loader)))
    # train_iter = iter(train_loader)
    # print(next(train_iter).keys())

    # train_set = CWQRankerDatset(kb_data_path='../data/demo_cwq_train_kb_cand.pkl',
    #                             text_data_path='../data/demo_cwq_train_text_cand.pkl',
    #                             train=True, num_kb_samples=20, num_text_samples=20)

    train_set = HotpotQARankerDataset(text_data_path='../data/demo_hotpotqa_train_text_cand.pkl',
                                      kb_data_path='../data/demo_hotpotqa_train_kb_cand.pkl',
                                      train=True, num_kb_samples=5, num_text_samples=5)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    for batch in train_loader:
        print(batch.keys())
        print(batch)
        exit()

    # train_text_set = HotpotQADataset('../data/demo_processed_text_train.pkl', train=True, num_samples=20)
    # dev_text_set = HotpotQADataset('../data/demo_processed_text_dev.pkl', train=False)

    # train_text_loader = DataLoader(train_text_set, batch_size=2, shuffle=True, num_workers=0)
    # dev_text_loader = DataLoader(dev_text_set, batch_size=1, shuffle=False, num_workers=0)

    # for batch in dev_text_loader:
    #     # print(batch['decomp_gold_paths'][0][0])
    #     a = [x[0] for x in batch['decomp_cand_paths'][0]]
    #     # print(batch['decomp_gold_paths'][0] in a)
    #     if batch['decomp_gold_paths'][0][0] in a:
    #         print(batch['decomp_gold_paths'][0][0])
    #         print(batch['question'], '-' * 10)
