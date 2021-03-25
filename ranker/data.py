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

class CWQDatset(Dataset):
    def __init__(self, data_path, train=False, num_samples=20):
        self.data = pickle.load(open(data_path, 'rb'))
        self.train = train
        self.num_samples = num_samples

        self.questions = [x['question'].strip() for x in self.data]

        decomps = [x['decomps'] for x in self.data]
        self.type = []
        self.decompositions = []
        for line in decomps:
            splits = line.split(' # ')
            if not len(splits) == 3:
                splits = handle_corner_case(line)
            self.type.append(splits[0])
            self.decompositions.append(splits[1:] + ['done'])

        assert (len(self.questions) == len(self.decompositions))
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.train:
            # get the paths
            hop1_gold = self.data[idx]['hop1_gold'][0]
            hop2_gold = self.data[idx]['hop2_gold'][0]
            all_hop1_cands = [x[0] for x in self.data[idx]['hop1_cands']]
            all_hop2_cands = [x[0] for x in self.data[idx]['hop2_cands']]
            # print(all_hop2_cands[:100])

            # sample negative samples
            num_neg_sampls = self.num_samples - 1
            if len(all_hop1_cands) >= num_neg_sampls:
                hop1_cands = random.sample(all_hop1_cands, k=num_neg_sampls)
            else:
                hop1_cands = all_hop1_cands
                if all_hop1_cands != []:
                    hop1_cands += random.choices(all_hop1_cands, k=num_neg_sampls - len(hop1_cands)) # all_hop1_cands may be []
                else:
                    hop1_cands = ['']*num_neg_sampls
                    

            if len(all_hop2_cands) >= num_neg_sampls:
                hop2_cands = random.sample(all_hop2_cands, k=num_neg_sampls)
            else:
                hop2_cands = all_hop2_cands
                if all_hop2_cands != []:
                    hop2_cands += random.choices(all_hop2_cands, k=num_neg_sampls - len(hop2_cands))
                else:
                    hop2_cands = ['']*num_neg_sampls

            label1 = np.random.randint(self.num_samples)
            hop1_cands.insert(label1, hop1_gold)

            label2 = np.random.randint(self.num_samples)
            hop2_cands.insert(label2, hop2_gold)

            cands = [hop1_cands, hop2_cands]
            gold_paths = [hop1_gold, hop2_gold]
            labels = [label1, label2]

            # sample negative samples as candidate reasoning path
            cand_reason_paths = []

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
                    count_part_2 += 1
            
            assert len(cand_reason_paths) == num_neg_sampls
            
            reason_path_label = np.random.randint(self.num_samples)
            cand_reason_paths.insert(reason_path_label, str(hop1_gold)+' '+str(hop2_gold))

            return {
                'question': self.questions[idx],
                "decompositions": self.decompositions[idx],
                'decomp_cand_paths': cands,
                'decomp_cand_labels': labels,
                'decomp_gold_paths': gold_paths,
                'cand_reason_paths': cand_reason_paths,
                'reason_path_label': reason_path_label
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


class HotpotQADataset(Dataset):
    def __init__(self, data_path, train=False, num_samples=20):
        self.train = train
        self.num_samples = num_samples
        self.data = pickle.load(open(data_path, 'rb'))
        self.len = len(self.data)
        self.questions = [x['question'].strip() for x in self.data]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # print(self.decomp_cand_labels[idx])
        if self.train:
            hop1_gold = self.data[idx]['hop1_gold'][1]  # ['title', 'paragraph']
            hop2_gold = self.data[idx]['hop2_gold'][1]
            all_hop1_cands = [x[1] for x in self.data[idx]['hop1_cands']]
            all_hop2_cands = [x[1] for x in self.data[idx]['hop2_cands']]
            gold_reason_path = hop1_gold+' '+hop2_gold
            # sample negative samples
            num_neg_sampls = self.num_samples - 1

            if len(all_hop1_cands) >= num_neg_sampls:
                hop1_cands = random.sample(all_hop1_cands, k=num_neg_sampls)
            else:
                hop1_cands = all_hop1_cands
                while len(hop1_cands) < num_neg_sampls:
                    try:
                        hop1_cands.append(self.data[random.randint(0, self.len-1)]['hop1_cands'][0][1])
                    except:
                        continue
                    
            if len(all_hop2_cands) >= num_neg_sampls:
                hop2_cands = random.sample(all_hop2_cands, k=num_neg_sampls)
            else:
                hop2_cands = all_hop2_cands
                while len(hop2_cands) < num_neg_sampls:
                    try:
                        hop2_cands.append(self.data[random.randint(0, self.len-1)]['hop2_cands'][0][1])
                    except:
                        continue

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

            # print('Train cand_reason_paths length: ', len(cand_reason_paths))
            # prepare candidates and lables used in retriever
            label1 = np.random.randint(self.num_samples)
            hop1_cands.insert(label1, hop1_gold)

            label2 = np.random.randint(self.num_samples)
            hop2_cands.insert(label2, hop2_gold)

            cands = [hop1_cands, hop2_cands]
            gold_paths = [hop1_gold, hop2_gold]
            labels = [label1, label2]

            # prepare paragraph for reader
            reader_paragraph = str(hop1_gold)+' '+str(hop2_gold)
            reader_paragraph = add_yes_no(reader_paragraph)
            # prepare start and end position
            start_position, end_position = find_span(reader_paragraph, answer)


            return {
                'id': self.data[idx]['id'],
                'question': self.questions[idx],
                'answer': self.data[idx]['answer'],
                'decomp_cand_paths': cands,
                'decomp_gold_paths': gold_paths,
                'decomp_cand_labels': labels,
                'cand_reason_paths': cand_reason_paths,
                'reason_path_label': reason_path_label,
                'reader_paragraph': reader_paragraph,
                'start_position': start_position,
                'end_position': end_position
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

    train_text_set = HotpotQADataset('../data/demo_processed_text_train.pkl', train=True, num_samples=20)
    dev_text_set = HotpotQADataset('../data/demo_processed_text_dev.pkl', train=False)

    train_text_loader = DataLoader(train_text_set, batch_size=2, shuffle=True, num_workers=0)
    dev_text_loader = DataLoader(dev_text_set, batch_size=1, shuffle=False, num_workers=0)

    for batch in dev_text_loader:
        # print(batch['decomp_gold_paths'][0][0])
        a = [x[0] for x in batch['decomp_cand_paths'][0]]
        # print(batch['decomp_gold_paths'][0] in a)
        if batch['decomp_gold_paths'][0][0] in a:
            print(batch['decomp_gold_paths'][0][0])
            print(batch['question'], '-' * 10)

        # print(len(a))
        # # print(a[:10])
        # print(a[0])
        # print(a[1])
        # print(a[2])
        # print(batch['decomp_gold_paths'][0][0])
        # print(batch['decomp_gold_paths'][0] in a)
        # exit()

    # for batch in train_loader:
    #     print(batch['question'])
    #     print(batch['decomp_cand_labels'])
    #     print(batch['decomp_cand_paths'])
    #     print(batch['decomp_gold_paths'])
    #     print(len(batch['decomp_cand_paths']))
    #     print(len(batch['decomp_cand_paths'][0]))
    #     print(len(batch['decomp_cand_paths'][0][0]))
    #     exit()
    # for batch in dev_loader:
    #     print(len(batch['decomp_cand_paths'][0][0]))
    #     print(batch['decomp_cand_paths'][0][1])
    #     print(len(batch['decomp_cand_aws'][0]))
    #     print(len(batch['decomp_cand_aws'][1]))
    #     exit()

    # pdb.set_trace()
