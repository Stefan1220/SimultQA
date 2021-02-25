import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers import PretrainedConfig, PreTrainedModel
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from generation_utils import GenerationUtils

import copy
import pdb
sys.path.append('../kb_candidate/')
from cand_path_retrieve_merge import retrieve_cand_path
sys.path.append('../text_candidate/bertserini/')
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
from retrieve_wikipedia import retrieve_hyper_linked_para
from retrieval_paragraph import retrieve_cand_paragraph

class MyModelConfig(PretrainedConfig):
    model_type = "encoder_decoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = BertConfig.from_pretrained(
            'bert-base-uncased', bos_token_id=101, eos_token_id=102)
        self.encoder.is_decoder = False
        self.encoder.return_dict = True
        self.encoder.hidden_dropout_prob = 0

        self.decoder = BertConfig.from_pretrained(
            'bert-base-uncased', bos_token_id=101, eos_token_id=102)
        self.decoder.is_decoder = True
        self.decoder.add_cross_attention = True
        self.decoder.is_encoder_decoder = True
        self.decoder.return_dict = True
        self.decoder.hidden_dropout_prob = 0

        self.is_encoder_decoder = True
        self.hidden_size = 768
        self.max_num_decomps = 3
        self.hidden_dropout_prob = 0.1

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class ModelRetriever(PreTrainedModel):
    config_class = MyModelConfig
    base_model_prefix = "encoder_decoder"

    def __init__(self, args):
        self.args = args
        self.config = config = MyModelConfig()
        super().__init__(config)
        self.encoder = BertModel.from_pretrained(
            'bert-base-uncased', config=config.encoder)
        self.decoder = BertLMHeadModel.from_pretrained(
            'bert-base-uncased', config=config.decoder)
        self.decoder.encoder = self.encoder
        self.decoder.get_encoder = lambda: self.encoder

        # RNN weight
        self.s = Parameter(torch.FloatTensor(
            config.hidden_size).uniform_(-0.1, 0.1))
        self.g = Parameter(torch.FloatTensor(1).fill_(1.0))
        self.rw = nn.Linear(2*config.hidden_size, config.hidden_size)

        self.bias = Parameter(torch.FloatTensor(1).zero_())
        self.qdw = nn.Linear(2*config.hidden_size, config.hidden_size)

        self.generation_utils = GenerationUtils(self.decoder)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.candidate_net = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(config.hidden_size, 1))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def weight_norm(self, state):
        state = state / state.norm(dim=-1, keepdim=True)
        # state = state / state.norm(dim=2).unsqueeze(2)
        state = self.g * state
        return state

    def cand_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        question_input_ids=None,
        question_mask_ids=None,
        question_type_ids=None,
        decomps_input_ids=None,
        decomps_mask_ids=None,
        decomps_type_ids=None,
        cand_input_ids=None,
        cand_mask_ids=None,
        cand_type_ids=None,
        text_labels=None,
        return_dict=None,
        **kwargs,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = text_labels.size(0)
        hidden_size = self.config.hidden_size
        # encode subqueries
        cand_para_encoding = self.encoder(
            input_ids=cand_input_ids,
            attention_mask=cand_mask_ids,
            token_type_ids=cand_type_ids,
            return_dict=True)

        cand_para_repr = cand_para_encoding[1]  # [B*N, L->1, D]
        # print(cand_para_repr.shape)
        cand_para_repr = cand_para_repr.reshape(
            batch_size, -1, hidden_size)
        num_cands = cand_para_repr.shape[1]

        # Hierarcical decomposition
        state = self.s.expand(batch_size, 1, self.s.size(0))
        state = self.weight_norm(state)
        states = []
        for i in range(2):
            if i == 0:
                h = state
            else:
                input = cand_para_repr[:, i-1:i, :]  # [B, 1, D]
                state = torch.cat((state, input), dim=2)  # (B, 1, 2*D)
                state = self.rw(state)  # (B, 1, D)
                state = self.weight_norm(state)
            states.append(state)

        states = torch.cat(states, dim=1)
        states = self.dropout(states)
        output = torch.bmm(states, cand_para_repr.transpose(1, 2)) # [B, 2, D], [B, D, N] -> [B, 2, N]
        output = output + self.bias # [B, 2, N]

        text_loss = F.binary_cross_entropy_with_logits(output, text_labels, reduction='mean')

        # QD loss
        '''
        states  # [B, 2, D]
        question_token_repr  # [B, L, D]
        decomps_token_repr  # [B*2, L, D]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids[:, i, :],
            attention_mask=decoder_attention_mask[:, i, :],
            encoder_hidden_states=qtr,
            encoder_attention_mask=attention_mask,
            labels=decoder_input_ids[:, i, :],
            return_dict=True)
        qd_loss = decoder_outputs['loss']
        '''
        # return {'qd_loss': torch.sum(torch.stack(qd_loss)), 'kbqa_loss': torch.sum(torch.stack(kbqa_loss))}
        return {'text_loss': text_loss}

    def chunking(self, input_dict, encoder, split_chunk=100):
        TOTAL = input_dict['input_ids'].size(0) # 500
        start = 0
        while start < TOTAL:
            end = min(start + split_chunk - 1, TOTAL - 1)
            chunk_len = end - start + 1
            input_ids_ = input_dict['input_ids'][start: start + chunk_len, :]
            attention_mask_ = input_dict['attention_mask'][start: start + chunk_len, :]
            token_type_ids_ = input_dict['token_type_ids'][start: start + chunk_len, :]

            cand_encoding_chunk = encoder(
                input_ids=input_ids_,
                attention_mask=attention_mask_,
                token_type_ids=token_type_ids_,
                return_dict=True) # decomp_cand_encoding_chunk[0].shape = [100,512,768] ; decomp_cand_encoding_chunk[1].shape = [100, 768]

            if start == 0:
                cand_encoding = cand_encoding_chunk[1]
            else:
                cand_encoding = torch.cat((cand_encoding, cand_encoding_chunk[1]), dim=0)

            start = end + 1

        cand_encoding = cand_encoding.contiguous()
        
        return cand_encoding

    def text_beam_search_inference(
        self,
        initial_hop=None,
        tokenizer=None,
        infer_type=None,
        save_size=10,
        return_dict=None,
    ):
        device = initial_hop['input_ids'].device
        # batch size is 1
        batch_size = 1
        hidden_size = self.config.hidden_size
        max_num_decomps = 2

        # encode subqueries
        decomp_cand_encoding = self.chunking(initial_hop, self.encoder) # [N, D]

        decomp_cand_repr = decomp_cand_encoding # [N, D]
        num_cands = decomp_cand_repr.shape[0]

        state = self.s.expand(batch_size, self.s.size(0))
        state = self.weight_norm(state) # state = [1, D]
        rnn_outputs = []
        hop_cand_scores = []
        hop_cand_paths = []

        h = state
        cur_cands = decomp_cand_repr # cur_cands = [N, D]
        cand_scores = torch.sigmoid( torch.bmm(state.unsqueeze(1), cur_cands.transpose(0,1).unsqueeze(0) ) + self.bias).squeeze() # [N,]

        cand_scores = cand_scores.detach().cpu().numpy() if device != 'cpu' else cand_scores.detach().numpy()
        cand_index = np.argsort(-cand_scores)  # select the best answer and path vector
        select_indexes = cand_index #[0:save_size]

        K = 3 # set k = 3
        topk_hop1_titles = [] 
        topk_hop1_paras = []
        for select_index in select_indexes[0:save_size]:
            topk_hop1_titles.append(initial_hop['cand_titles'][select_index])
            topk_hop1_paras.append(initial_hop['cand_paragraphs'][select_index])
        #'''
        outputs = {}
        cand_paras_merge = []
        cand_paras_unmerge = []
        cand_titles_unmerge = []
        decomp_gold_repr_list = torch.empty(len(select_indexes), hidden_size)
        hop1_indexes = []
        for i, select_index in enumerate(select_indexes):
            if len(hop1_indexes) >= save_size:
                break
            decomp_gold_repr_ = cur_cands[select_index]
            select_para = initial_hop['cand_paragraphs'][select_index]
            question = initial_hop['question'][0]
            first_hop_title = initial_hop['cand_titles'][select_index]

            if infer_type.startswith('htqa'):
                cand_title_paras = retrieve_hyper_linked_para(first_hop_title) 
                # add top-ranked paragraphs from the first hop into candidates of the second hop 
                count = 0
                for j, (title, para) in enumerate(zip(topk_hop1_titles, topk_hop1_paras)):
                    if count >= K:
                        break
                    if j != i:
                        cand_title_paras.append([title, para])
                        count += 1
            else: # cwq
                cand_title_paras = retrieve_cand_paragraph(question, select_para, 10)

            if len(cand_title_paras) == 0:
                continue

            cand_paras = [ (question, x[1]) for x in cand_title_paras ]
            cand_paras_only = [x[1] for x in cand_title_paras ]
            cand_title = [x[0] for x in cand_title_paras]

            cand_paras_merge.extend(cand_paras)
            cand_paras_unmerge.append(cand_paras_only)    
            cand_titles_unmerge.append(cand_title)  
            decomp_gold_repr_list[i] = decomp_gold_repr_
            hop1_indexes.append(select_index)

        decomp_gold_repr_list = decomp_gold_repr_list[0:len(hop1_indexes), :].to(device)

        cand_para_id = tokenizer.batch_encode_plus(
            cand_paras_merge, add_special_tokens=True, padding=True, return_tensors='pt', truncation=True
        )
        second_hop = {
            'input_ids': cand_para_id['input_ids'].to(device), 
            'attention_mask': cand_para_id['attention_mask'].to(device),
            'token_type_ids': cand_para_id['token_type_ids'].to(device)
        }

        decomp_cand_encoding_ = self.chunking(second_hop, self.encoder) # [N, D]

        decomp_cand_repr_ = decomp_cand_encoding_ # [N, D]
        num_cands_ = decomp_cand_repr_.shape[0]
        input_ = decomp_gold_repr_list # [M, D]
        state_ = torch.cat((state.repeat(input_.shape[0],1), input_), dim=-1) # [M, 2D]
        state_ = self.rw(state_) # [M, D]
        state_ = self.weight_norm(state_) # [M, D]
        cur_cands_ = decomp_cand_repr_ # [N, D]
        cand_scores_ = torch.sigmoid( torch.bmm( state_.unsqueeze(0), cur_cands_.transpose(0,1).unsqueeze(0) ) + self.bias ).squeeze() # cand_scores_ = [M, N]

        outputs['hop1_cand'] = [ initial_hop['cand_paragraphs'][idx] for idx in  hop1_indexes]
        outputs['hop1_title'] = [ initial_hop['cand_titles'][idx] for idx in  hop1_indexes]                   
        outputs['hop1_score'] = [ cand_scores[idx] for idx in hop1_indexes ]
        outputs['hop2_cand'] = cand_paras_unmerge
        outputs['hop2_title'] = cand_titles_unmerge

        hop2_scores = []
        start = 0
        for i in range(len(cand_paras_unmerge)):
            j = len(cand_paras_unmerge[i])
            hop2_scores.append(cand_scores_[i, start:start+j].tolist())
            start = start + j
        outputs['hop2_score'] = hop2_scores

        return outputs

    def kb_beam_search_inference(
        self,
        initial_hop=None,
        tokenizer=None,
        save_size=5,
        return_dict=None,
    ):

        device = initial_hop['input_ids'].device
        # batch size is 1
        batch_size = 1
        hidden_size = self.config.hidden_size
        max_num_decomps = 2

        # encode subqueries
        decomp_cand_encoding = self.chunking(initial_hop, self.encoder) # [N, D]

        decomp_cand_repr = decomp_cand_encoding # [N, D]
        num_cands = decomp_cand_repr.shape[0]

        state = self.s.expand(batch_size, self.s.size(0))
        state = self.weight_norm(state) # state = [1, D]
        rnn_outputs = []
        hop_cand_scores = []
        hop_cand_paths = []

        h = state
        cur_cands = decomp_cand_repr # cur_cands = [N, D]
        cand_scores = torch.sigmoid( torch.bmm(state.unsqueeze(1), cur_cands.transpose(0,1).unsqueeze(0) ) + self.bias).squeeze() # [N,]

        cand_scores = cand_scores.detach().cpu().numpy() if device != 'cpu' else cand_scores.detach().numpy()
        cand_index = np.argsort(-cand_scores)  # select the best answer and path vector
        select_indexes = cand_index
        
        outputs = {}
        cand_path_merge = []
        cand_path_unmerge = []
        cand_ans_unmerge = []
        decomp_gold_repr_list = torch.empty(len(select_indexes), hidden_size)
        hop1_indexes = []
        for i, select_index in enumerate(select_indexes):
            if len(hop1_indexes) >= save_size:
                break
            question = initial_hop['question'][0]
            decomp_gold_repr_ = cur_cands[select_index]
            topic_ents = initial_hop['cand_ans'][select_index]
            topic_ents = [x[0] for x in topic_ents]
            try:
                if initial_hop['class_entity']['entity_cons'] == []:
                    ner_entities = []
                else:
                    ner_entities = [ x[0] for x in initial_hop['class_entity']['entity_cons'] ]
                path_plus_aws = retrieve_cand_path('hop2', topic_ents, nte_list=[], cons_e_list=ner_entities, type_e=initial_hop['class_entity']['entity_type'][0], year=initial_hop['class_entity']['year'][0], superlative=initial_hop['class_entity']['superlative_word'][0])
            except Exception as e:
                print('Error in retrieve_cand_path: ', e)
                continue

            # cand_path = [ question+' '+x[0] for x in path_plus_aws]
            cand_path = [ (question, x[0]) for x in path_plus_aws ]
            cand_path_only = [x[0] for x in path_plus_aws]
            cand_ans_only = [x[1] for x in path_plus_aws]
            if len(cand_path) == 0:
                continue

            cand_path_merge.extend(cand_path)
            cand_path_unmerge.append(cand_path_only)    
            cand_ans_unmerge.append(cand_ans_only)  

            decomp_gold_repr_list[i] = decomp_gold_repr_
            hop1_indexes.append(select_index)

        decomp_gold_repr_list = decomp_gold_repr_list[0:len(hop1_indexes), :].to(device)

        cand_path_id = tokenizer.batch_encode_plus(
            cand_path_merge, add_special_tokens=True, padding=True, return_tensors='pt', truncation=True
        )
        second_hop = {
            'input_ids': cand_path_id['input_ids'].to(device), 
            'attention_mask': cand_path_id['attention_mask'].to(device),
            'token_type_ids': cand_path_id['token_type_ids'].to(device)
        }

        decomp_cand_encoding_ = self.chunking(second_hop, self.encoder) # [N, D]

        decomp_cand_repr_ = decomp_cand_encoding_ # [N, D]
        num_cands_ = decomp_cand_repr_.shape[0]
        input_ = decomp_gold_repr_list # [M, D]
        state_ = torch.cat((state.repeat(input_.shape[0],1), input_), dim=-1) # [M, 2D]
        state_ = self.rw(state_) # [M, D]
        state_ = self.weight_norm(state_) # [M, D]
        cur_cands_ = decomp_cand_repr_ # [N, D]
        cand_scores_ = torch.sigmoid( torch.bmm( state_.unsqueeze(0), cur_cands_.transpose(0,1).unsqueeze(0) ) + self.bias ).squeeze() # cand_scores_ = [M, N]

        outputs['hop1_cand'] = [ initial_hop['cand_paths'][idx] for idx in  hop1_indexes]
        outputs['hop1_ans'] = [ initial_hop['cand_ans'][idx] for idx in  hop1_indexes]                   
        outputs['hop1_score'] = [ cand_scores[idx] for idx in hop1_indexes ]
        outputs['hop2_cand'] = cand_path_unmerge
        outputs['hop2_ans'] = cand_ans_unmerge

        hop2_scores = []
        start = 0
        for i in range(len(cand_path_unmerge)):
            j = len(cand_path_unmerge[i])
            hop2_scores.append(cand_scores_[i, start:start+j].tolist())
            start = start + j
        outputs['hop2_score'] = hop2_scores

        return outputs

    # def inference(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     return_dict=None
    # ):
    #     # Encode complex question
    #     question_encoding = self.encoder(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         return_dict=True)
    #     question_token_repr = question_encoding['last_hidden_state']
    #     batch_size = question_token_repr.size(0)
    #
    #     # Hierarcical decomposition
    #     state = self.s.expand(batch_size, 1, self.s.size(0))
    #     state = self.weight_norm(state)
    #     rnn_outputs = []
    #     subquestions = []
    #     decomposition_repr = None
    #     for i in range(self.config.max_num_decomps):
    #         if i == 0:
    #             h = state
    #         else:
    #             # RNN output and state
    #             input = decomposition_repr  # (B, 1, D)
    #             state = torch.cat((state, input), dim=2)  # (B, 1, 2*D)
    #             state = self.rw(state)  # (B, 1, D)
    #             state = self.weight_norm(state)
    #             rnn_outputs.append(state)
    #
    #         qtr = self.qdw(torch.cat(
    #                 [question_token_repr, state.expand(-1, question_token_repr.size(1), -1)],
    #                 dim=2))
    #         # pdb.set_trace()
    #
    #         # Decompose sub-questions
    #         subquestion = self.generation_utils.generate(
    #             encoder_hidden_states=qtr,
    #             attention_mask=attention_mask,
    #             max_length=20,
    #             min_length=1,
    #             num_beams=4,
    #             repetition_penalty=1.0,
    #             bos_token_id=101,
    #             eos_token_id=102
    #         )
    #         subquestions.append(subquestion)
    #
    #         # Encode generated sub-questions
    #         end = torch.tensor(
    #             [self.decoder.config.eos_token_id] * subquestion.size(0), dtype=torch.int64).unsqueeze(1)
    #         subquestion = torch.cat([subquestion, end], 1)
    #         subquestion_mask = self.get_mask_of_generated_sequence(subquestion)
    #         decomposition_encoding = self.encoder(
    #             input_ids=subquestion,
    #             attention_mask=subquestion_mask,
    #             return_dict=True)
    #         decomposition_repr = decomposition_encoding.last_hidden_state[:,0:1,:]
    #
    #     return subquestions

    def get_mask_of_generated_sequence(self, input_ids):
        batch_size, input_length = input_ids.shape
        found_eos = torch.tensor([0] * batch_size, dtype=torch.bool)
        res = []
        for i in range(input_length):
            found_eos = torch.logical_or(
                found_eos, (input_ids[:, i] == self.decoder.config.eos_token_id))
            mask = torch.logical_or(
                found_eos, (input_ids[:, i] == self.decoder.config.pad_token_id))
            res.append(torch.logical_not(mask))
        return torch.stack(res, 1).type(torch.int64)

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, encoder_outputs, **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

        # Ideally all models should have a `use_cache`
        # leave following to ifs until all have it implemented
        if "use_cache" in decoder_inputs:
            input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

        if "past_key_values" in decoder_inputs:
            input_dict["past_key_values"] = decoder_inputs["past_key_values"]

        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


if __name__ == '__main__':

    model = MyModel()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Toy data
    batch = {
        "question": [
            "what state is home to the university that is represented in sports by george washington colonials men 's basketball",
            "what year did the team with baltimore fight song win the superbowl",
            "which school with the fight song '' the orange and blue `` did emmitt smith play for"],
        "decompositions": [
            ["the education institution has a sports team named george washington colonials men 's basketball",
             "what state is the %composition",
             "done"
             ],
            ["the sports team with the fight song the baltimore fight song",
             "what year did %composition win the superbowl",
             "done"
             ],
            ["what football teams did emmitt smith play for",
             "what football teams is the sports team with the fight song the orange and blue",
             "done"
             ]]
    }

    question = tokenizer.batch_encode_plus(
        batch["question"], add_special_tokens=True, padding=True, return_tensors='pt')
    decompositions = tokenizer.batch_encode_plus(sum(
        batch["decompositions"], []), add_special_tokens=True, padding=True, return_tensors='pt')

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for _ in range(10):
        outputs = model(
            input_ids=question['input_ids'],
            token_type_ids=question['token_type_ids'],
            attention_mask=question['attention_mask'],
            decoder_input_ids=decompositions['input_ids'],
            decoder_token_type_ids=decompositions['token_type_ids'],
            decoder_attention_mask=decompositions['attention_mask'],
            # decoder_num_hops=[len(_)
            #                   for _ in batch["decompositions"]],
            return_dict=True)
        qd_loss = outputs['qd_loss']
        print(f"Epoch {_:3d}: qd_loss {qd_loss.data:2.4f}")

        optimizer.zero_grad()
        qd_loss.backward()
        optimizer.step()

    # TODO: save and load from pretrained
    # # model.save_pretrained("bert2bert")
    # # model = MyModel.from_pretrained("bert2bert")
    # pdb.set_trace()
    torch.save(model.state_dict, "bert2bert/model.pt")
    model = MyModel()
    model.load_state_dict("bert2bert/model.pt")

    # Generation
    generated = model.inference(
        input_ids=question['input_ids'],
        token_type_ids=question['token_type_ids'],
        attention_mask=question['attention_mask'])
    # pdb.set_trace()
    print(generated)
    for batch_idx in range(3):
        print(batch_idx)
        for i in range(len(generated)):
            print(tokenizer.decode(generated[i][batch_idx]))
