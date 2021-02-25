import os
import sys
import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast

from retriever_model import ModelRetriever
from data import *
from utils import *

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]

def inference(args, model, dev_set, modality, save_path):

    print('Dev Length: ', len(dev_set))
    dev_loader = DataLoader(
        dev_set, batch_size=1, shuffle=False, num_workers=0)
    
    model.to(args.device)
    model.eval()
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    with torch.no_grad():
        with open(save_path, 'w') as f:
            for step, batch in enumerate(tqdm(dev_loader, disable=args.close_tqdm)):
                print(step)
                data_dict = dict()
                data_dict['question'] = batch['question'][0]

                if modality == 'kb':
                    cand_paths_only = [x[0][0] for x in batch['hop1_cand'] ]
                    cand_paths = [ (batch["question"][0],  x[0][0]) for x in batch['hop1_cand'] ]
                    cand_ans = [ x[1] for x in batch['hop1_cand'] ] # x[1] = [(mid, ), (mid, ), ...]
                    try:
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

                elif modality == 'text':
                    data_dict['answer'] = batch['answer'][0]
                    cand_paragraphs_only = [x[0] for x in batch['initial_candidates'] ]
                    cand_paragraphs = [ (batch["question"][0],  x[0]) for x in batch['initial_candidates'] ]
                    cand_titles = [ x[0] for x in batch['initial_titles'] ]
                    
                    try:
                        cand_para_ids = tokenizer.batch_encode_plus(
                            cand_paragraphs, add_special_tokens=True, padding=True, 
                            return_tensors='pt', truncation=True, max_length=args.max_sent_len)

                        initial_hop_dict = {
                            'input_ids': cand_para_ids['input_ids'].to(args.device),
                            'token_type_ids': cand_para_ids['token_type_ids'].to(args.device),
                            'attention_mask': cand_para_ids['attention_mask'].to(args.device),
                            'cand_paragraphs': cand_paragraphs_only,
                            'question': batch['question'],
                            'cand_titles': cand_titles
                        }

                        outputs = model.text_beam_search_inference(
                            initial_hop = initial_hop_dict,
                            tokenizer = tokenizer,
                            infer_type = args.infer_type,
                        )

                        data_dict['hop1_cand'] = outputs['hop1_cand']
                        data_dict['hop1_title'] = outputs['hop1_title']                   
                        data_dict['hop1_score'] = outputs['hop1_score']
                        data_dict['hop2_cand'] = outputs['hop2_cand']
                        data_dict['hop2_score'] = outputs['hop2_score']
                        data_dict['hop2_title'] = outputs['hop2_title']   
                    except Exception as e:
                        print('Error: ', e)
                        data_dict['hop1_cand'] = []
                        data_dict['hop1_title'] = []                  
                        data_dict['hop1_score'] = []
                        data_dict['hop2_cand'] = []
                        data_dict['hop2_score'] = []
                        data_dict['hop2_title'] = []  
         
                f.write(str(data_dict) + '\n')

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
    parser.add_argument('--train_text_dir', type=str, default='../data/')
    parser.add_argument('--train_kb_dir', type=str, default='../data/')
    parser.add_argument('--dev_data_dir', type=str, default='../data/')

    parser.add_argument('--infer_type', type=str, default='cwq_kb')

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

    # load model
    model = ModelRetriever(args).to(args.device)

    model_path = '/data/mo.169/CQD4QA/85/HybridQA/saved_models/hybrid_negcands_bm_20_hy_10_kb_30/saved_pytorch_model_model_epoch_2.pt'
    tmp_model_state_dict = torch.load(model_path)
    # nn.DataParallel wraps the model => change the keys of parameters when loading model
    model_state_dict = {key.replace("module.", ""): value for key, value in tmp_model_state_dict.items()}
    model.load_state_dict(model_state_dict)

    infer_type = args.infer_type

    # Candidate paths
    if infer_type == 'cwq_kb':
        modality = 'kb'
        cwq_kb_path = '/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/cwq_kb_dev.txt'
        save_path = '/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/cwq_kb.txt'
        dev_set = CWQDataset(
            cwq_kb_path,
            '/data/mo.169/CQD4QA/85/HybridQA/data/dev-src.txt',
            '/data/mo.169/CQD4QA/85/HybridQA/data/dev-tgt.txt',
            train=False,
            num_kb_cands=args.num_kb_cands,
            chunk_size=args.chunk_size
        )

    elif infer_type == 'cwq_text':
        modality = 'text'
        cwq_text_path = '/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/cwq_text_dev.txt' 
        save_path = '/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/cwq_text.txt'
        dev_set = HotpotQADataset(
            cwq_text_path,
            train=False
        )

    elif infer_type == 'htqa_kb':
        modality = 'kb'
        htqa_kb_path = '/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/htqa_kb_dev.txt'
        save_path = '/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/htqa_kb.txt'
        dev_set = CWQDataset(
            htqa_kb_path,
            '', # dev-src.txt
            '', # dev-tgt.txt
            train=False,
            num_kb_cands=args.num_kb_cands,
            chunk_size=args.chunk_size
        )

    elif infer_type == 'htqa_text':
        modality = 'text'
        htqa_text_path = '/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/htqa_text_dev.txt' 
        save_path = '/data/mo.169/CQD4QA/85/HybridQA/saved_data/retriever/cross_data_no_qd/htqa_text.txt'
        dev_set = HotpotQADataset(
            htqa_text_path,
            train=False
        )
    
    inference(args, model, dev_set, modality, save_path)
    
if __name__ == '__main__':
    main()
