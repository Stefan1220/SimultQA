'''
API for reader
'''
from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling_reader import BertForQuestionAnsweringConfidence
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from rc_utils import convert_examples_to_features_yes_no, read_squad_examples, write_predictions_yes_no_no_empty_answer, write_predictions_yes_no_beam

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])

# parameters
bert_model = "bert-large-uncased"
output_dir = "/data/mo.169/ICLR2020/models/reader/"
max_seq_length = 384
doc_stride = 128
max_query_length = 64
do_train = False
do_predict = True
predict_batch_size = 1
n_best_size = 20
max_answer_length = 30
verbose_logging = False
no_cuda = False
seed = 42
do_lower_case = True
local_rank = -1
version_2_with_negative = False
null_score_diff_threshold = 0.0
no_masking = False
skip_negatives = False

model = BertForQuestionAnsweringConfidence.from_pretrained(
    output_dir,  num_labels=4, no_masking=no_masking)
tokenizer = BertTokenizer.from_pretrained(
    output_dir, do_lower_case=do_lower_case)

def reader_predict(context):

    if local_rank == -1 or no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.to(device)
    
    # NOTE: predict answer
    if do_predict and (local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=context, is_training=False, version_2_with_negative=version_2_with_negative,
            max_answer_len=max_answer_length, skip_negatives=skip_negatives)
        eval_features = convert_examples_to_features_yes_no(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False)

        # logger.info("***** Running predictions *****")
        # logger.info("  Num orig examples = %d", len(eval_examples))
        # logger.info("  Num split examples = %d", len(eval_features))
        # logger.info("  Batch size = %d", predict_batch_size)

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=local_rank not in [-1, 0]):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_switch_logits = model(
                    input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                switch_logits = batch_switch_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             switch_logits=switch_logits))
        output_prediction_file = os.path.join(
            output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            output_dir, "nbest_predictions")
        output_null_log_odds_file = os.path.join(
            output_dir, "null_odds")
        
        prediction = write_predictions_yes_no_no_empty_answer(eval_examples, eval_features, all_results,
                                                 n_best_size, max_answer_length,
                                                 do_lower_case, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file, verbose_logging,
                                                 version_2_with_negative, null_score_diff_threshold,
                                                 no_masking)
        return prediction


if __name__ == "__main__":
    main()
