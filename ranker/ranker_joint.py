'''
Based on the text-only training checkpoint, 
continue to train the ranker by KB modality
'''
import os
import sys
import json
import argparse
import numpy as np
import copy
import pdb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers import PretrainedConfig, PreTrainedModel
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel, BertTokenizerFast
from data import CWQDatset, HotpotQADataset

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]

class Ranker(nn.Module):
    def __init__(self, args, hid):
        super(Ranker, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = hid

        self.candidate_net = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))

    def forward(
            self,
            cand_path_input_ids=None,
            cand_path_token_type_ids=None,
            cand_path_attention_mask=None,
            reason_paths_labels=None,
            batch_size=None,
            return_dict=None,
    ):

        hidden_size = self.hidden_size
        # encode reasoning paths

        # cand_encoding = self.encoder(
        #     input_ids=cand_path_input_ids, # [B*N, L]
        #     attention_mask=cand_path_attention_mask,
        #     token_type_ids=cand_path_token_type_ids,
        #     return_dict=True)
        
        dic = {
            'input_ids':cand_path_input_ids, # [B*N, L]
            'attention_mask':cand_path_attention_mask,
            'token_type_ids':cand_path_token_type_ids,  
        }

        cand_encoding = self.chunking(dic, self.encoder)

        cand_repr = cand_encoding  # [B*N, L->1, D]

        cand_repr = cand_repr.reshape(
            batch_size, -1, hidden_size) # [B, N, D], N = 100(the number of candidates for each hop
        num_cands = cand_repr.shape[1]

        cand_scores = self.candidate_net(cand_repr).squeeze(2)  # [B, N, 1]

        if reason_paths_labels is None: # for testing
            loss = 0
        else: # for training
            loss = F.cross_entropy(cand_scores, reason_paths_labels)

        return {'loss': loss, 'cand_scores': cand_scores}

    def chunking(self, hop, encoder):
        TOTAL = hop['input_ids'].size(0)
        start = 0
        split_chunk = 50
        while start < TOTAL:
            end = min(start+split_chunk-1, TOTAL-1)
            chunk_len = end-start+1
            input_ids_ = hop['input_ids'][start:start+chunk_len, :]
            attention_mask_ = hop['attention_mask'][start:start+chunk_len, :]
            token_type_ids_ = hop['token_type_ids'][start:start+chunk_len, :]

            decomp_cand_encoding_chunk = encoder(
                input_ids=input_ids_,
                attention_mask=attention_mask_,
                token_type_ids=token_type_ids_,
                return_dict=True) # decomp_cand_encoding_chunk[0].shape = [100,512,768] ; decomp_cand_encoding_chunk[1].shape = [100, 768]
            if start == 0:
                decomp_cand_encoding = decomp_cand_encoding_chunk[1]
            else:
                decomp_cand_encoding = torch.cat((decomp_cand_encoding, decomp_cand_encoding_chunk[1]), dim=0)
            start = end + 1

        decomp_cand_encoding = decomp_cand_encoding.contiguous()
        
        return decomp_cand_encoding

def train(args, batch, model, tokenizer, optimizer):
    torch.set_grad_enabled(True)
    model.train(True)
    batch_size = len(batch['question'])
    cand_reason_paths = [] # don't contain yes no
    for bi in range(len(batch['cand_reason_paths'][0])):
        for ni in range(len(batch['cand_reason_paths'])):
            # cand_reason_paths.append( batch['question'][0] + ' ' + str(batch['cand_reason_paths'][ni][bi]) )
            cand_reason_paths.append( (batch['question'][bi], str(batch['cand_reason_paths'][ni][bi]) ))

    cand_reason_labels = torch.LongTensor(batch['reason_path_label'])

    cand_reason_paths_ids = tokenizer.batch_encode_plus(
        cand_reason_paths, add_special_tokens=True, padding=True, return_tensors='pt', truncation=True)

    outputs = model(
        cand_path_input_ids=cand_reason_paths_ids['input_ids'].to(args.device),
        cand_path_token_type_ids=cand_reason_paths_ids['token_type_ids'].to(args.device),
        cand_path_attention_mask=cand_reason_paths_ids['attention_mask'].to(args.device),
        reason_paths_labels=cand_reason_labels.to(args.device),
        batch_size=batch_size,
        return_dict=True)

    loss = outputs['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {'loss': loss.item()}

def evaluate(args, data_loader, num_iter,  model, tokenizer):
    torch.set_grad_enabled(False)
    model.train(False)
    dev_iter = iter(data_loader)
    rank_list = []

    for i in range(num_iter):
        print(i)
        batch, dev_iter = get_iter_batch(dev_iter, data_loader)

        cand_reason_paths = []
        for bi in range(len(batch['cand_reason_paths'][0])):
            for ni in range(len(batch['cand_reason_paths'])):
                cand_reason_paths.append( (batch['question'][bi], str(batch['cand_reason_paths'][ni][bi]) ))

        cand_reason_labels = torch.LongTensor(batch['reason_path_label'])

        cand_reason_paths_ids = tokenizer.batch_encode_plus(
            cand_reason_paths, add_special_tokens=True, padding=True, return_tensors='pt', truncation=True)

        outputs = model(
            cand_path_input_ids=cand_reason_paths_ids['input_ids'].to(args.device),
            cand_path_token_type_ids=cand_reason_paths_ids['token_type_ids'].to(args.device),
            cand_path_attention_mask=cand_reason_paths_ids['attention_mask'].to(args.device),
            reason_paths_labels=cand_reason_labels.to(args.device),
            batch_size=1,
            return_dict=True)
        
        cand_scores = outputs['cand_scores'][0]

        temp_label = np.zeros(cand_scores.shape[0])
        temp_label[cand_reason_labels[0]] = 1
        score_ans = [(cand_scores[j], temp_label[j]) for j in range(cand_scores.shape[0])]
        score_ans.sort(key=lambda x: x[0], reverse=True)
        rank_list.append([x[1] for x in score_ans])

    mrr = mean_reciprocal_rank(rank_list)
    
    return mrr
    

def main():

    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--close_tqdm', type='bool', default=True)
    parser.add_argument('--kb_batch_size', type=int, default=5)
    parser.add_argument('--kb_dev_batch_size', type=int, default=1)
    parser.add_argument('--text_batch_size', type=int, default=1)
    parser.add_argument('--text_dev_batch_size', type=int, default=1)
    parser.add_argument('--max_num_hops', type=int, default=2)
    parser.add_argument('--num_kb_cands', type=int, default=30)
    parser.add_argument('--num_text_cands', type=int, default=30) 
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=100)
    args = parser.parse_args()

    # Load CWQ Data
    train_kb_set = CWQDatset('/home/mo.169/Projects/CQD4QA/prepare_data/data/processed_kb_train.pkl', train=True, num_samples=args.num_kb_cands)
    dev_kb_set = CWQDatset('/home/mo.169/Projects/CQD4QA/prepare_data/data/processed_kb_dev.pkl', train=False, num_samples=args.num_kb_cands)
    train_kb_loader = DataLoader(train_kb_set, batch_size=args.kb_batch_size, shuffle=True, num_workers=0)
    dev_kb_loader = DataLoader(dev_kb_set, batch_size=args.kb_dev_batch_size, shuffle=False, num_workers=0)

    # Load HotpotQA Data
    train_text_set = HotpotQADataset('/home/mo.169/Projects/CQD4QA/prepare_data/data/processed_text_train.pkl', train=True, num_samples=args.num_text_cands)
    dev_text_set = HotpotQADataset('/home/mo.169/Projects/CQD4QA/prepare_data/data/processed_text_dev.pkl', train=False, num_samples=args.num_text_cands)
    train_text_loader = DataLoader(train_text_set, batch_size=args.text_batch_size, shuffle=True, num_workers=0)
    dev_text_loader = DataLoader(dev_text_set, batch_size=1, shuffle=False, num_workers=0)

    # Build model
    model = Ranker(args, hid=768)
    # path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/v1/model_epoch_0_step_70000.pt'
    # model.load_state_dict(torch.load(path))
    # Freeze the first 9 layers of BERT
    modules = [model.encoder.embeddings, model.encoder.encoder.layer[:9]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    device = torch.device("cuda:0")
    args.device = device
    model.to(device)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_iter_kb = train_kb_set.len // args.kb_batch_size
    num_iter_text = train_text_set.len // args.text_batch_size
    num_iter = max(num_iter_kb, num_iter_text)
    print('num_iter: ', num_iter)
    for epoch in range(args.epochs): 
        step = 0
        kb_sum_loss = []
        text_sum_loss = []
        train_kb_iter = iter(train_kb_loader)
        train_text_iter = iter(train_text_loader)
        with tqdm(total=num_iter, desc=f'Epoch {epoch}/{args.epochs}', unit='batch', disable=args.close_tqdm) as pbar:
            for i in range(num_iter):
                try:
                    cur_kb_batch, train_kb_iter = get_iter_batch(train_kb_iter, train_kb_loader)
                    kb_train_metrics = train(args, cur_kb_batch, model, tokenizer, optimizer)
                    kb_sum_loss.append(kb_train_metrics['loss'])

                    cur_text_batch, train_text_iter = get_iter_batch(train_text_iter, train_text_loader)
                    text_train_metrics = train(args, cur_text_batch, model, tokenizer, optimizer)
                    text_sum_loss.append(text_train_metrics['loss'])

                    pbar.update(1)
                    step += 1
                    pbar.set_postfix(**{'kb loss': kb_train_metrics['loss'], 'text loss': text_train_metrics['loss']})

                    if step % args.print_interval == 0:
                        print('Epoch={}\tstep={}\tKB Loss={}\tText Loss={}'.format(
                            epoch, step, np.mean(kb_sum_loss), np.mean(text_sum_loss)
                        ))
                        kb_sum_loss = []
                        text_sum_loss = []

                    if step % args.eval_interval == 0:
                        save_path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_%s_step_%s.pt'%(epoch, step)
                        torch.save(model.state_dict(), save_path)
                except Exception as e:
                    print(e)
                    continue

        # Save model for each epoch
        save_path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_%s.pt'%(epoch)
        torch.save(model.state_dict(), save_path)

    # # load model and evaluate
    path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_1.pt'
    model.load_state_dict(torch.load(path))
    num_iter1 = dev_kb_set.len
    mrr1 = evaluate(args, dev_kb_loader, num_iter1,  model, tokenizer)
    print('CWQ MRR: ', mrr1)
    num_iter2 = dev_text_set.len
    mrr2 = evaluate(args, dev_text_loader, num_iter2,  model, tokenizer)
    print('HotpotQA MRR: ', mrr2)


if __name__ == '__main__':
    main()