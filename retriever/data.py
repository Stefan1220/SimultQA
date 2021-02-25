import re
import pdb
import json
import random
import pickle
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


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


class HotpotQADataset(Dataset):
    def __init__(self, data_path, 
                 train=False, 
                 num_bm_cands=10,
                 num_hy_cands=10, 
                 chunk_size=4):

        self.train = train
        self.num_bm_cands = num_bm_cands
        self.num_hy_cands = num_hy_cands
        self.chunk_size = chunk_size

        #tmp_start = 1518

        if data_path[-3::] == 'txt':
            self.data = [eval(x) for x in open(data_path).readlines()]#[tmp_start:]
        elif data_path[-3::] == 'pkl':
            self.data = pickle.load(open(data_path, 'rb'))#[tmp_start:]

        self.size = len(self.data)
        # self.questions = [x['question'].strip() for x in self.data]

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # print(self.decomp_cand_labels[idx])
        if self.train:
            # hop1_gold = self.data[idx]['supporting_facts_hop1'][1][:15]  # ['title', 'paragraph']
            # hop2_gold = self.data[idx]['supporting_facts_hop2'][1][:15]
            # bm_candidates = [x[1][:15] for x in self.data[idx]['bm_candidates']]
            # hy_candidates = [x[1][:15] for x in self.data[idx]['hyperlink_candidates']]

            hop1_title = self.data[idx]['supporting_facts_hop1'][0]
            hop2_title = self.data[idx]['supporting_facts_hop2'][0]
            # bm_cand_titles = [x[0] for x in self.data[idx]['bm_candidates']]
            # hy_cand_titles = [x[0] for x in self.data[idx]['hyperlink_candidates']]

            hop1_gold = self.data[idx]['supporting_facts_hop1'][1]  # ['title', 'paragraph']
            hop2_gold = self.data[idx]['supporting_facts_hop2'][1]
            if 'bm_candidates' in self.data[0]:
                bm_candidates = [x[1] for x in self.data[idx]['bm_candidates'] 
                    if x[0] != hop1_title and x[0] != hop2_title and x[1] != None and x[1] != [] and x[1] != '']
            else:
                bm_candidates = [x[1] for x in self.data[idx]['tfidf_candidates'] 
                    if x[0] != hop1_title and x[0] != hop2_title and x[1] != None and x[1] != [] and x[1] != '']

            hy_candidates = [x[1] for x in self.data[idx]['hyperlink_candidates'] 
                if x[0] != hop1_title and x[0] != hop2_title and x[1] != None and x[1] != [] and x[1] != '']

            # assert len(bm_candidates) > 0
            # assert len(hy_candidates) > 0
            # sample or not?
            # bm_cands = random.sample(bm_candidates, k=self.num_bm_cands)
            # hy_cands = random.sample(hy_candidates, k=self.num_hy_cands)
            if len(bm_candidates) >= self.num_bm_cands:
                bm_cands = bm_candidates[:self.num_bm_cands]
            else:
                bm_cands = bm_candidates + random.choices(bm_candidates, k=self.num_bm_cands-len(bm_candidates))
            
            if len(hy_candidates) >= self.num_hy_cands:
                hy_cands = hy_candidates[:self.num_hy_cands]
            else:
                hy_cands = hy_candidates + random.choices(hy_candidates, k=self.num_hy_cands-len(hy_candidates))

            assert len(bm_cands) > 0
            assert len(hy_cands) > 0

            chunk_paras = []
            chunk_labels = []
            neg_cands = bm_cands + hy_cands
            neg_chunk_size = self.chunk_size - 2
            num_chunks = (len(neg_cands) // neg_chunk_size) + 1
            for i in range(num_chunks):
                neg_chunk = neg_cands[i * neg_chunk_size: (i + 1) * neg_chunk_size]
                if i == num_chunks - 1:
                    neg_chunk = neg_cands[i * neg_chunk_size::]
                if len(neg_chunk) == 0:
                    continue

                cur_chunk = [hop1_gold, hop2_gold] + neg_chunk
                cur_label = np.array([[1, 0] + [0] * len(neg_chunk), 
                                      [0, 1] + [0] * len(neg_chunk)])
                # print(len(cur_chunk))
 
                chunk_paras.append(cur_chunk)
                chunk_labels.append(cur_label)

            question = self.data[idx]['question']
            # print(chunk_paras)

            return {
                'question': question,
                'cand_paragraphs': chunk_paras,
                'labels': chunk_labels
            }

        else:
            hop1_gold = self.data[idx]['supporting_facts_hop1'][1]  # ['title', 'paragraph']
            hop2_gold = self.data[idx]['supporting_facts_hop2'][1]

            hop1_title = self.data[idx]['supporting_facts_hop1'][0]  # ['title', 'paragraph']
            hop2_title = self.data[idx]['supporting_facts_hop2'][0]

            bm_cand_titles = [x[0] for x in self.data[idx]['bm_candidates']]
            bm_candidates = [x[1] if x[1] is not None else '' for x in self.data[idx]['bm_candidates'] ] # sometimes the paragraph is None
            question = self.data[idx]['question']
            answer = self.data[idx]['answer']

            return {
                'question': question,
                'answer': answer,
                'hop1_gold': hop1_gold,
                'hop2_gold': hop2_gold,
                'hop1_title': hop1_title,
                'hop2_title': hop2_title,
                'initial_titles': bm_cand_titles,
                'initial_candidates': bm_candidates,               
            }


class CWQDataset(Dataset):
    def __init__(self, cand_data_path,
                 src_data_path, 
                 tgt_data_path,
                 train=False, 
                 num_kb_cands=10, 
                 chunk_size=4):

        self.train = train
        self.num_kb_cands = num_kb_cands
        self.chunk_size = chunk_size

        if cand_data_path[-3::] == 'txt':
            self.data = [eval(x) for x in open(cand_data_path).readlines()]
        elif cand_data_path[-3::] == 'pkl':
            self.data = pickle.load(open(cand_data_path, 'rb'))

        if src_data_path != '':
            self.dev_class_entity = [ eval(x) for x in open('/data/mo.169/CQD4QA/85/HybridQA/data/cwq_dev_class_entity_no_assume_superlative.txt').readlines() ]

            data_src = open(src_data_path).readlines()
            data_tgt = open(tgt_data_path).readlines()

            self.questions = [x.strip() for x in data_src]
            decomps = [x.strip() for x in data_tgt]
            self.type = []
            self.decompositions = []
            for line in decomps:
                splits = line.split(' # ')
                if not len(splits) == 3:
                    splits = handle_corner_case(line)
                self.type.append(splits[0])
                self.decompositions.append(splits[1:] + ['done'])
            self.type = self.type
            self.decompositions = self.decompositions
        else:
            self.questions = [ eval(x)['question'] for x in open('/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/htqa_text_dev.txt').readlines() ]
            self.dev_class_entity = [{'entity_1':[], 'entity_2':[], 'intermediate_ans':[], 'entity_type':'', 'entity_cons':[], 'year':'', 'superlative_word':''}]*len(self.data)
            self.decompositions = ['']*len(self.data)

        if self.train:
            remove_idx = set([i for i, x in enumerate(self.data) 
                if len(set([xx[0] for xx in x['cand_hop1']] + [xx[0] for xx in x['cand_hop2']])) == 0 
                or x['gold_hop1'][0] == [] or x['gold_hop2'][0] == []])

            self.data = [x for i, x in enumerate(self.data) if i not in remove_idx]
            self.questions = [x for i, x in enumerate(self.questions) if i not in remove_idx]
            self.decompositions = [x for i, x in enumerate(self.decompositions) if i not in remove_idx]

        assert (len(self.questions) == len(self.decompositions))
        self.size = len(self.data)
        # self.questions = [x['question'].strip() for x in self.data]

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # print(self.decomp_cand_labels[idx])
        if self.train:
            hop1_gold = self.data[idx]['gold_hop1'][0]  # [path, [mids]]
            hop2_gold = self.data[idx]['gold_hop2'][0]
            
            all_hop1_cands = [x[0] for x in self.data[idx]['cand_hop1']]
            all_hop2_cands = [x[0] for x in self.data[idx]['cand_hop2']]
            all_cands = list(set(all_hop1_cands + all_hop2_cands))

            if len(all_cands) >= self.num_kb_cands:
                neg_cands = random.sample(all_cands, k=self.num_kb_cands)
            else:
                neg_cands = all_cands + random.choices(all_cands, k=self.num_kb_cands-len(all_cands))
            
            # if len(all_hop2_cands) >= self.num_kb_cands:
            #     hop2_cands = random.sample(all_hop2_cands, k=self.num_kb_cands)
            # else:
            #     hop2_cands = all_hop2_cands + random.choices(all_hop2_cands, k=self.num_kb_cands-len(all_hop2_cands))

            chunk_paras = []
            chunk_labels = []
            # neg_cands = hop1_cands + hop2_cands
            neg_chunk_size = self.chunk_size - 2
            num_chunks = (len(neg_cands) // neg_chunk_size) + 1
            for i in range(num_chunks):
                neg_chunk = neg_cands[i * neg_chunk_size: (i + 1) * neg_chunk_size]
                if i == num_chunks - 1:
                    neg_chunk = neg_cands[i * neg_chunk_size::]
                if len(neg_chunk) == 0:
                    continue

                cur_chunk = [hop1_gold, hop2_gold] + neg_chunk
                cur_label = np.array([[1, 0] + [0] * len(neg_chunk), 
                                      [0, 1] + [0] * len(neg_chunk)])
                # print(len(cur_chunk))
 
                chunk_paras.append(cur_chunk)
                chunk_labels.append(cur_label)

            return {
                'question': self.questions[idx],
                'decomps': self.decompositions[idx],
                'cand_paragraphs': chunk_paras,
                'labels': chunk_labels
            }

        else:
            try:
                hop1_gold = self.data[idx]['gold_hop1']  # [path, [mid]]
                hop2_gold = self.data[idx]['gold_hop2']
            except KeyError:
                hop1_gold = ['', ['']]
                hop2_gold = ['', ['']]

            all_cand_hop1 = self.data[idx]['cand_hop1'] # [path, [mid]]*N
            all_cand_hop2 = self.data[idx]['cand_hop2']

            # answer = self.data[idx]['answer'] # [mid1, mid2, ...]

            return {
                'hop1_gold': hop1_gold,
                'hop2_gold': hop2_gold,
                'hop1_cand': all_cand_hop1,
                'hop2_cand': all_cand_hop2,
                'question': self.questions[idx],
                'class_entity': self.dev_class_entity[idx],          
            }

if __name__ == '__main__':
    train_text_set = HotpotQADataset(
        '../data/cand_paragraphs_iclr_tfidf_hyperlink.pkl',
        train=True,
        num_bm_cands=20,
        num_hy_cands=10,
        chunk_size=10
    )
    train_text_loader = DataLoader(
        train_text_set, batch_size=2, 
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    for batch in tqdm(train_text_set):
        # print([len(x) for x in batch['cand_paragraphs']])
        for chunk_idx in range(len(batch['cand_paragraphs'])):
            for bi in range(len(batch['cand_paragraphs'][chunk_idx][0])):
                for mi in range(len(batch['cand_paragraphs'][chunk_idx])):
                    if len(batch['cand_paragraphs'][chunk_idx][mi]) == 0:
                        print(batch['cand_paragraphs'][chunk_idx][mi])
                        exit()

    # train_set = CWQDataset(
    #     '../data/cwq_train_cand_paths.pkl',
    #     '../data/train-src.txt',
    #     '../data/train-tgt.txt',
    #     train=True,
    #     num_kb_cands=10,
    #     chunk_size=5
    # )
    # train_loader = DataLoader(
    #     train_set, batch_size=3, 
    #     shuffle=True, num_workers=0, drop_last=True)

    # for batch in train_loader:
    #     print(batch)
    #     exit()

    '''
    train_set = HotpotQADataset(
        '../data/demo_10_cand_paragraphs_tfidf_hyperlink.txt',
        train=True,
        num_bm_cands=40,
        num_hy_cands=10,
        chunk_size=8
        )

    train_loader = DataLoader(
        train_set, batch_size=3, shuffle=False, num_workers=0)
    # dev_loader = DataLoader(
    #     dev_set, batch_size=4, shuffle=False, num_workers=0)

    try:
        for batch in tqdm(train_loader):
            pass
    except Exception as e:
        print(e)
        # print(batch)
    '''
        # cand_paras = []
        # for bi in range(len(batch['cand_paragraphs'][0])):
        #     for mi in range(len(batch['cand_paragraphs'])):
        #         cand_paras.append(batch["question"][bi] + ' ' + batch['cand_paragraphs'][mi][bi])
        # print(cand_paras)

    # pdb.set_trace()
