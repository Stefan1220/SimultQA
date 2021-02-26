import os
import sys
import pdb
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertTokenizerFast

from retriever_model import ModelRetriever
from data import *
# from evals import *
from utils import *
# from train import *
# from inference import *
# from fuzzywuzzy import fuzz

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]

def main():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_sent_len', type=int, default=378)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('-c', '--close_tqdm', type='bool', default=True)
    parser.add_argument('--print_step_interval', type=int, default=100)
    parser.add_argument('--save_step_interval', type=int, default=1000)
    # parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='0', help='seperate the number by comma without any whitespace')

    parser.add_argument('--max_num_hops', type=int, default=2)
    parser.add_argument('-bm', '--num_bm_cands', type=int, default=40)
    parser.add_argument('-hy', '--num_hy_cands', type=int, default=10)
    parser.add_argument('--num_kb_cands', type=int, default=50)
    parser.add_argument('-cs', '--chunk_size', type=int, default=6, help='total #paras in one pass including two golden paras')
    parser.add_argument('--grad_acc_steps', type=int, default=1)

    parser.add_argument('--train', type='bool', default=False)
    parser.add_argument('--output_dir', type=str, default='../saved_models')
    parser.add_argument('--output_suffix', type=str, default='')
    parser.add_argument('--train_text_dir', type=str, default='../data/cand_paragraphs_tfidf_hyperlink.txt')
    parser.add_argument('--train_kb_dir', type=str, default='../data/cand_paragraphs_tfidf_hyperlink.txt')
    parser.add_argument('--dev_data_dir', type=str, default='../data/cwq_dev_cand_paths_no_assume_1_25.txt')

    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = True

    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_gpu = torch.cuda.device_count()

    if args.train:
        train_kb_set = CWQDataset(
            args.train_kb_dir,
            '../data/train-src.txt',
            '../data/train-tgt.txt',
            train=True,
            num_kb_cands=args.num_kb_cands,
            chunk_size=args.chunk_size
        )
        train_text_set = HotpotQADataset(
            # '../data/demo_cand_paragraphs_bm_hyperlink.txt',
            # '../data/cand_paragraphs_bm_hyperlink.txt',
            args.train_text_dir,
            train=True,
            num_bm_cands=args.num_bm_cands,
            num_hy_cands=args.num_hy_cands,
            chunk_size=args.chunk_size
        )
        train_kb_loader = DataLoader(
            train_kb_set, batch_size=args.train_batch_size, 
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        train_text_loader = DataLoader(
            train_text_set, batch_size=args.train_batch_size, 
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        # Build model
        model = ModelRetriever(args).to(args.device)

        # model_path = '../saved_models/tfidf_negcands_bm_40_hy_10/saved_pytorch_model_model_epoch_2.pt'
        # tmp_model_state_dict = torch.load(model_path)
        # # nn.DataParallel wraps the model => change the keys of parameters when loading model
        # model_state_dict = {key.replace("module.", ""): value for key, value in tmp_model_state_dict.items()}
        # model.load_state_dict(model_state_dict)
        # print('Model Loaded!')

        if args.num_gpu > 1:
            model = nn.DataParallel(model)

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Training
        total_size = train_text_set.size + train_kb_set.size
        num_training_steps = int(total_size / args.train_batch_size / args.grad_acc_steps * args.epochs)
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
        model.train()
        optimizer.zero_grad()

        num_iter = total_size // args.train_batch_size
        for epoch in range(args.epochs):
            # step = 0
            sum_loss = []
            train_kb_iter = iter(train_kb_loader)
            train_text_iter = iter(train_text_loader)
            with tqdm(total=num_iter, desc=f'Epoch {epoch}/{args.epochs}', unit='sample',
                    disable=args.close_tqdm) as pbar:
                # for step, batch in enumerate(train_loader):
                for step in range(num_iter):
                    if np.random.rand() > 0.5:
                        cur_kb_batch, train_kb_iter = get_iter_batch(train_kb_iter, train_kb_loader)
                        batch = cur_kb_batch
                    else:
                        cur_text_batch, train_text_iter = get_iter_batch(train_text_iter, train_text_loader)
                        batch = cur_text_batch
                    '''
                    question_ids = tokenizer.batch_encode_plus(
                        batch["question"], add_special_tokens=True, padding='max_length', 
                        return_tensors='pt', truncation=True, max_length=args.max_sent_len)

                    decomps = []
                    for bi in range(len(batch['decomps'])):
                        for hi in range(2):  # only 2 hops
                            decomps.append(batch['decomps'][hi][bi])

                    decomps_ids = tokenizer.batch_encode_plus(
                        decomps, add_special_tokens=True, padding='max_length', 
                        return_tensors='pt', truncation=True, max_length=args.max_sent_len)                    
                    '''
                    # negative chunking
                    num_chunks = len(batch['cand_paragraphs'])
                    batch_loss = []
                    for chunk_idx in range(num_chunks):

                        cand_paragraphs = []
                        for bi in range(len(batch['cand_paragraphs'][chunk_idx][0])):
                            for mi in range(len(batch['cand_paragraphs'][chunk_idx])):
                                # concatenate question with each candidate paragraph
                                try:
                                    cand_paragraphs.append((batch["question"][bi], batch['cand_paragraphs'][chunk_idx][mi][bi]))
                                except Exception as e:
                                    print(e)
                                    print(batch['cand_paragraphs'][chunk_idx])
                                    exit()
                        
                        cand_para_ids = tokenizer.batch_encode_plus(cand_paragraphs,
                            add_special_tokens=True, padding=True, return_tensors='pt', 
                            truncation=True, max_length=args.max_sent_len)

                        cand_labels = batch['labels'][chunk_idx].to(args.device, dtype=torch.float)
                        # print(len(cand_paragraphs), cand_labels.shape, cand_para_ids['input_ids'].shape)
                        
                        outputs = model(
                            # question_input_ids=question_ids['input_ids'].to(args.device),
                            # question_mask_ids=question_ids['attention_mask'].to(args.device),
                            # question_type_ids=question_ids['token_type_ids'].to(args.device),
                            # decomps_input_ids=decomps_ids['input_ids'].to(args.device),
                            # decomps_mask_ids=decomps_ids['attention_mask'].to(args.device),
                            # decomps_type_ids=decomps_ids['token_type_ids'].to(args.device),
                            cand_input_ids=cand_para_ids['input_ids'].to(args.device),
                            cand_mask_ids=cand_para_ids['attention_mask'].to(args.device),
                            cand_type_ids=cand_para_ids['token_type_ids'].to(args.device),
                            text_labels=cand_labels,
                            return_dict=True)

                        loss = outputs['text_loss']
                        # print(loss)
                        
                        if args.num_gpu > 1:
                            loss = loss.mean()

                        if args.grad_acc_steps > 1:
                                loss = loss / args.grad_acc_steps

                        loss.backward()
                        batch_loss.append(loss.item())

                    # step += 1
                    sum_loss.append(np.mean(batch_loss))

                    if (step + 1) % args.grad_acc_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    pbar.set_postfix(**{'loss (batch)': np.mean(batch_loss)})
                    pbar.update(len(batch["question"]))

                    if (step + 1) % args.print_step_interval == 0:
                        print('Epoch={}\tStep={}\tTrain loss={:.4f} at {}'.format(epoch, step, 
                            np.mean(sum_loss), datetime.now().strftime("%m/%d/%Y %X")))
                        sum_loss = []

                    if (step + 1) % args.save_step_interval == 0:
                        output_dir = os.path.join(args.output_dir, 
                            '{}_negcands_bm_{}_hy_{}_kb_{}'.format(args.output_suffix, 
                                args.num_bm_cands, args.num_hy_cands, args.num_kb_cands))
                        save(model, output_dir, 'model_epoch_{}_step_{}'.format(epoch, step))

                # save the model at the end of each epoch
                output_dir = os.path.join(args.output_dir, 
                    '{}_negcands_bm_{}_hy_{}_kb_{}'.format(args.output_suffix, 
                    args.num_bm_cands, args.num_hy_cands, args.num_kb_cands))
                save(model, output_dir, 'model_epoch_{}'.format(epoch))
                    
    else:  # evaluation
        dev_set = CWQDataset(
            args.dev_data_dir,
            '../data/dev-src.txt',
            '../data/dev-tgt.txt',
            train=False,
            num_kb_cands=args.num_kb_cands,
            chunk_size=args.chunk_size
        )
        print('Dev Length: ', len(dev_set))
        dev_loader = DataLoader(
            dev_set, batch_size=1, shuffle=False, num_workers=0)

        # load model
        model = ModelRetriever(args).to(args.device)

        model_path = '../saved_models/kb_negcands_kb_20/saved_pytorch_model_model_epoch_2.pt'
        tmp_model_state_dict = torch.load(model_path)
        # nn.DataParallel wraps the model => change the keys of parameters when loading model
        model_state_dict = {key.replace("module.", ""): value for key, value in tmp_model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        model.eval()
        
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        with torch.no_grad():
            with open('../saved_data/retriever/inference_kb_20_epoch_2_with_class_entity_no_assume_1_25.txt', 'w') as f:
                for step, batch in enumerate(tqdm(dev_loader, disable=args.close_tqdm)):
                    # if step >= 50:
                    #     break
                    print(step)
                    data_dict = dict()
                    data_dict['question'] = batch['question'][0]

                    cand_paths_only = [x[0][0] for x in batch['hop1_cand'] ]
                    cand_paths = [ (batch["question"][0],  x[0][0]) for x in batch['hop1_cand'] ]
                    cand_ans = [ x[1] for x in batch['hop1_cand'] ] # x[1] = [(mid, ), (mid, ), ...]
                    try:
                        # "tokenizer.batch_encode_plus" takes much more time
                        cand_path_ids = tokenizer.batch_encode_plus(
                            cand_paths, add_special_tokens=True, padding=True, 
                            return_tensors='pt', truncation=True, max_length=args.max_sent_len)

                        initial_hop_dict = {
                            'input_ids': cand_path_ids['input_ids'].to(args.device),
                            'token_type_ids': cand_path_ids['token_type_ids'].to(args.device),
                            'attention_mask': cand_path_ids['attention_mask'].to(args.device),
                            'cand_paths': cand_paths_only,
                            'question': batch['question'],
                            'cand_ans': cand_ans,
                            'class_entity': batch['class_entity'],
                        }

                        outputs = model.kb_beam_search_inference(
                            initial_hop = initial_hop_dict,
                            tokenizer = tokenizer
                        )

                        data_dict['hop1_cand'] = outputs['hop1_cand']
                        data_dict['hop1_ans'] = outputs['hop1_ans']                   
                        data_dict['hop1_score'] = outputs['hop1_score']
                        data_dict['hop2_cand'] = outputs['hop2_cand']
                        data_dict['hop2_score'] = outputs['hop2_score']
                        data_dict['hop2_ans'] = outputs['hop2_ans']
                    except Exception as e:
                        print('Error: ', e)
                        data_dict['hop1_cand'] = []
                        data_dict['hop1_ans'] = []                  
                        data_dict['hop1_score'] = []
                        data_dict['hop2_cand'] = []
                        data_dict['hop2_score'] = []
                        data_dict['hop2_ans'] = []                       

                    f.write(str(data_dict) + '\n')


def calculate_path_recall(infer_data, dev_loader, beam_size):

    count = 0
    path_hit = 0
    for i, batch in enumerate(dev_loader):
        if count >= len(infer_data):
            break

        gold_hop1_title = batch['hop1_title'][0]
        gold_hop2_title = batch['hop2_title'][0]
        infer_dict = eval(infer_data[i].strip())
        cand_reason_paths_score = []
        for j in range(len(infer_dict['hop1_title'][0:beam_size])): # set Beam
            path_score = [ [ [infer_dict['hop1_title'][j], title], infer_dict['hop1_score'][j] * score] for title, score in zip(infer_dict['hop2_title'][j], infer_dict['hop2_score'][j] if isinstance(infer_dict['hop2_score'][j], list) else [infer_dict['hop2_score'][j]] ) ]
            cand_reason_paths_score.extend(path_score)
        
        cand_reason_paths_score_sort = sorted(cand_reason_paths_score, key=lambda x:x[1], reverse=True)
        cand_reason_paths = [ x[0:1] for x in  cand_reason_paths_score_sort][0:beam_size]

        for cand_reason_path in cand_reason_paths:
            if cand_reason_path[0] == [gold_hop1_title, gold_hop2_title] or cand_reason_path[0] == [gold_hop2_title, gold_hop1_title]:
                path_hit += 1
                break
        count += 1
    print(path_hit, count, path_hit/count)

def calculate_hop1_recall(infer_data, dev_loader, beam_size):

    count = 0
    hop1_hit = 0
    for i, batch in enumerate(dev_loader):
        if count >= len(infer_data):
            break
        gold_hop1_title = batch['hop1_title'][0]
        infer_dict = eval(infer_data[i].strip())

        if gold_hop1_title in infer_dict['hop1_title'][0:beam_size]:
            hop1_hit += 1
        count += 1
    print(hop1_hit, count, hop1_hit/count)

def calculate_hop2_recall(infer_data, dev_loader, beam_size):

    count = 0
    hop2_hit = 0
    for i, batch in enumerate(dev_loader):
        if count >= len(infer_data):
            break
        gold_hop2_title = batch['hop2_title'][0]
        infer_dict = eval(infer_data[i].strip())

        for level_1 in infer_dict['hop2_title'][0:beam_size]:
            mark = False
            for level_2 in level_1:
                if level_2 == gold_hop2_title:
                    hop2_hit += 1
                    mark = True
                    break
            if mark:
                break
        count += 1
    print(hop2_hit, count, hop2_hit/count)


if __name__ == '__main__':
    main()
    '''
    file_name = 'inference_bm_40_hy_10_epoch_2.txt'

    # load inference data
    infer_data = open('../saved_data/retriever/%s'%(file_name), 'r').readlines()

    dev_set = HotpotQADataset('../data/cand_paragraphs_tfidf_500.txt',train=False)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)

    beam_size = 50
    calculate_path_recall(infer_data, dev_loader, beam_size)
    calculate_hop1_recall(infer_data, dev_loader, beam_size)
    calculate_hop2_recall(infer_data, dev_loader, beam_size)
    '''