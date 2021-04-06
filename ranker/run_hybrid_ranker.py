'''
Based on the text-only training checkpoint, 
continue to train the ranker by KB modality
'''
import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel, BertTokenizerFast

from ranker_loader import CWQRankerDatset, HotpotQARankerDataset


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def get_iter_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)

    return batch, data_iter


def save(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'saved_pytorch_model_{}.pt'.format(save_prefix))
    print('Save the model at {}'.format(save_path))
    torch.save(model.state_dict(), save_path)


class HybridRankerModel(nn.Module):
    def __init__(self, args, hid):
        super(HybridRankerModel, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = hid
        self.candidate_net = nn.Sequential(nn.Linear(self.hidden_size, 1))


    def loss(self, list_logits, list_labels):
        # [B, N]
        return -torch.sum(list_labels * F.log_softmax(list_logits, dim=1), 1).mean()

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
        batch_size = reason_paths_labels.shape[0]
        # encode reasoning paths

        cand_encoding = self.encoder(
            input_ids=cand_path_input_ids, # [B*N, L]
            attention_mask=cand_path_attention_mask,
            token_type_ids=cand_path_token_type_ids,
            return_dict=True)
        
        # dic = {
        #     'input_ids':cand_path_input_ids, # [B*N, L]
        #     'attention_mask':cand_path_attention_mask,
        #     'token_type_ids':cand_path_token_type_ids,  
        # }
        # cand_encoding = self.chunking(dic, self.encoder)

        cand_path_repr = cand_encoding[1]  # [B*N, L->1, D]

        cand_path_repr = cand_path_repr.reshape(
            batch_size, -1, hidden_size) # [B, N, D], N = 100(the number of candidates for each hop
        num_cands = cand_path_repr.shape[1]

        cand_path_scores = self.candidate_net(cand_path_repr).squeeze(2)  # [B, N, 1]

        if reason_paths_labels is None: # for testing
            loss = 0
        else: # for training
            try:
                loss = self.loss(cand_path_scores, reason_paths_labels)
            except Exception as e:
                print(e)
                print(cand_path_scores.shape, reason_paths_labels.shape)
                print(cand_path_repr.shape, cand_path_input_ids.shape)
                exit()

        return {'loss': loss, 'cand_scores': cand_path_scores}

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

def train(args, batch, model, tokenizer, optimizer, scheduler):
    torch.set_grad_enabled(True)
    model.train(True)

    batch_size = len(batch['question'])
    cand_reason_paths = [] # don't contain yes no
    for bi in range(len(batch['cand_reason_paths'][0])):
        for ni in range(len(batch['cand_reason_paths'])):
            cand_reason_paths.append((batch['question'][bi], str(batch['cand_reason_paths'][ni][bi])))

    # cand_reason_labels = torch.FloatTensor(batch['reason_path_label'])
    cand_reason_labels = batch['reason_path_label'].to(torch.float)

    cand_reason_paths_ids = tokenizer.batch_encode_plus(cand_reason_paths, 
        add_special_tokens=True, padding=True, return_tensors='pt', truncation=True, max_length=args.max_sent_len)

    outputs = model(
        cand_path_input_ids=cand_reason_paths_ids['input_ids'].to(args.device),
        cand_path_token_type_ids=cand_reason_paths_ids['token_type_ids'].to(args.device),
        cand_path_attention_mask=cand_reason_paths_ids['attention_mask'].to(args.device),
        reason_paths_labels=cand_reason_labels.to(args.device),
        batch_size=batch_size,
        return_dict=True)

    loss = outputs['loss']
    if args.num_gpu > 1:
        loss = loss.mean()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

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
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--close_tqdm', type='bool', default=True)
    parser.add_argument('--print_step_interval', type=int, default=100)
    parser.add_argument('--save_step_interval', type=int, default=1000)
    parser.add_argument('--train', type='bool', default=False)
    parser.add_argument('--max_sent_len', type=int, default=378)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--kb_batch_size', type=int, default=5)
    parser.add_argument('--kb_dev_batch_size', type=int, default=1)
    parser.add_argument('--text_batch_size', type=int, default=1)
    parser.add_argument('--text_dev_batch_size', type=int, default=1)
    parser.add_argument('--max_num_hops', type=int, default=2)
    parser.add_argument('--num_kb_cands', type=int, default=30)
    parser.add_argument('--num_text_cands', type=int, default=30)

    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=100)

    parser.add_argument('--cwq_text_path', type=str, default='')
    parser.add_argument('--cwq_kb_path', type=str, default='')
    parser.add_argument('--hpqa_text_path', type=str, default='')
    parser.add_argument('--hpqa_kb_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='../saved_models')
    parser.add_argument('--output_suffix', type=str, default='')

    parser.add_argument('--cwq_text_sample',type=int, default=10)
    parser.add_argument('--cwq_kb_sample',type=int, default=10)
    parser.add_argument('--hpqa_text_sample',type=int, default=10)
    parser.add_argument('--hpqa_kb_sample',type=int, default=10)

    parser.add_argument('--train_batch_size', type=int, default=5)

    args = parser.parse_args()
    print(args)

    # Load CWQ Data
    train_cwq_set = CWQRankerDatset(kb_data_path=args.cwq_kb_path,
                                    text_data_path=args.cwq_text_path,
                                    train=True, 
                                    num_kb_samples=args.cwq_kb_sample, 
                                    num_text_samples=args.cwq_text_sample)
    train_cwq_loader = DataLoader(train_cwq_set, 
                                  batch_size=args.train_batch_size, 
                                  shuffle=True, num_workers=0)
    # Load HotpotQA Data
    train_hpqa_set = HotpotQARankerDataset(text_data_path=args.hpqa_text_path,
                                           kb_data_path=args.hpqa_kb_path,
                                           train=True, 
                                           num_kb_samples=args.hpqa_kb_sample, 
                                           num_text_samples=args.hpqa_text_sample)
    train_hpqa_loader = DataLoader(train_hpqa_set, 
                                   batch_size=args.train_batch_size, 
                                   shuffle=True, num_workers=0)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = True
    
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_gpu = torch.cuda.device_count()

    # Build model
    model = HybridRankerModel(args, hid=768).to(args.device)
    # path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/v1/model_epoch_0_step_70000.pt'
    # model.load_state_dict(torch.load(path))

    # Freeze the first 9 layers of BERT
    modules = [model.encoder.embeddings, model.encoder.encoder.layer[:9]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    if args.num_gpu > 1:
        model = nn.DataParallel(model)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training
    total_size = train_cwq_set.len + train_hpqa_set.len
    num_training_steps = int(total_size / args.train_batch_size * args.epochs)
    num_warmup_steps = int(num_training_steps * 0.1)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)  # PyTorch scheduler

    num_iter = total_size // args.train_batch_size
    print('Begin training ...')
    for epoch in range(args.epochs): 
        step = 0
        cwq_sum_loss = []
        hpqa_sum_loss = []
        train_cwq_iter = iter(train_cwq_loader)
        train_hpqa_iter = iter(train_hpqa_loader)
        with tqdm(total=num_iter, desc=f'Epoch {epoch}/{args.epochs}', unit='batch', disable=args.close_tqdm) as pbar:
            for step in range(num_iter):
                if np.random.rand() > 0.5:
                    # batch, train_kb_iter = get_iter_batch(train_kb_iter, train_kb_loader)
                    cur_cwq_batch, train_cwq_iter = get_iter_batch(train_cwq_iter, train_cwq_loader)
                    cwq_train_metrics = train(args, cur_cwq_batch, model, tokenizer, optimizer, scheduler)
                    cwq_sum_loss.append(cwq_train_metrics['loss'])
                    loss = cwq_train_metrics['loss']
                else:
                    # batch, train_text_iter = get_iter_batch(train_text_iter, train_text_loader)
                    cur_hpqa_batch, train_hpqa_iter = get_iter_batch(train_hpqa_iter, train_hpqa_loader)
                    hpqa_train_metrics = train(args, cur_hpqa_batch, model, tokenizer, optimizer, scheduler)
                    hpqa_sum_loss.append(hpqa_train_metrics['loss'])
                    loss = hpqa_train_metrics['loss']

                # print(loss)

                if (step + 1) % args.print_step_interval == 0:
                    print('Epoch={}\tStep={} \tTrain CWQ loss={:.4f} HPQA loss={:.4f} Total loss={:.4f} at {}'.format(
                        epoch, step, np.mean(cwq_sum_loss), np.mean(hpqa_sum_loss), np.mean(cwq_sum_loss + hpqa_sum_loss),
                        datetime.now().strftime("%m/%d/%Y %X")))
                    cwq_sum_loss = []
                    hpqa_sum_loss = []

                if (step + 1) % args.save_step_interval == 0:
                    output_dir = os.path.join(args.output_dir, 
                        '{}_negs_cwq_text_{}_kb_{}_hpqa_text_{}_kb_{}'.format(args.output_suffix,
                                                                              args.cwq_text_sample, 
                                                                              args.cwq_kb_sample, 
                                                                              args.hpqa_text_sample,
                                                                              args.hpqa_kb_sample))
                    save(model, output_dir, 'epoch_{}_step_{}'.format(epoch, step))

        # save the model at the end of each epoch
        output_dir = os.path.join(args.output_dir, 
            '{}_negs_cwq_text_{}_kb_{}_hpqa_text_{}_kb_{}'.format(args.output_suffix,
                                                                    args.cwq_text_sample, 
                                                                    args.cwq_kb_sample, 
                                                                    args.hpqa_text_sample,
                                                                    args.hpqa_kb_sample))
        save(model, output_dir, 'epoch_{}'.format(epoch))

                # try:
                #     cur_kb_batch, train_kb_iter = get_iter_batch(train_kb_iter, train_kb_loader)
                #     kb_train_metrics = train(args, cur_kb_batch, model, tokenizer, optimizer)
                #     kb_sum_loss.append(kb_train_metrics['loss'])

                #     cur_text_batch, train_text_iter = get_iter_batch(train_text_iter, train_text_loader)
                #     text_train_metrics = train(args, cur_text_batch, model, tokenizer, optimizer)
                #     text_sum_loss.append(text_train_metrics['loss'])

                #     pbar.update(1)
                #     step += 1
                #     pbar.set_postfix(**{'kb loss': kb_train_metrics['loss'], 'text loss': text_train_metrics['loss']})

                #     if step % args.print_interval == 0:
                #         print('Epoch={}\tstep={}\tKB Loss={}\tText Loss={}'.format(
                #             epoch, step, np.mean(kb_sum_loss), np.mean(text_sum_loss)
                #         ))
                #         kb_sum_loss = []
                #         text_sum_loss = []

                #     if step % args.eval_interval == 0:
                #         save_path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_%s_step_%s.pt'%(epoch, step)
                #         torch.save(model.state_dict(), save_path)
                # except Exception as e:
                #     print(e)
                #     continue

        # # Save model for each epoch
        # save_path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_%s.pt'%(epoch)
        # torch.save(model.state_dict(), save_path)

    # # load model and evaluate
    # path = '/data/mo.169/CQD4QA/checkpoints/ranker/kb_text_joint/fix_bert/new_neg_sample/model_epoch_1.pt'
    # model.load_state_dict(torch.load(path))
    # num_iter1 = dev_kb_set.len
    # mrr1 = evaluate(args, dev_kb_loader, num_iter1,  model, tokenizer)
    # print('CWQ MRR: ', mrr1)
    # num_iter2 = dev_text_set.len
    # mrr2 = evaluate(args, dev_text_loader, num_iter2,  model, tokenizer)
    # print('HotpotQA MRR: ', mrr2)


if __name__ == '__main__':
    main()