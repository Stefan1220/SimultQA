import json
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
from fuzzywuzzy import fuzz

searcher = build_searcher("/data/mo.169/index/lucene-index.enwiki-20180701-paragraphs")

def load_data(path):
    # load hotpot qa dataset
    data = json.load(open(path))

    return data

def retrieve_cand_paragraph(question, paragraph, num):
    query = question if paragraph == '' else question + ' ' + paragraph
    q = Question(query)
    # fetch some contexts from Wikipedia with Pyserini
    contexts = retriever(q, searcher, num)

    candidate_facts = []
    for context in contexts:
        # print(context.text, ' ', context.score, '\n')
        candidate_facts.append(context.text.split(' . '))

    return candidate_facts

def evaluate(path): 

    with open(path, 'r') as f:
        data = f.readlines()

    total_score = 0
    total_score_hop1 = 0
    total_score_hop2 = 0
    for item in data:
        item = eval(item)
        supp_hop1 = item['supporting_facts_hop1']
        supp_hop2 = item['supporting_facts_hop2']
        cand_hop1 = item['candidate_facts_hop1']
        cand_hop2 = item['candidate_facts_hop2']

        cands1 = [item[1] for item in cand_hop1]
        for cand in cands1:
            score = fuzz.ratio(supp_hop1[1], cand)
            if score >= 80:
                total_score += 1
                total_score_hop1 += 1
                break
        # if supp_hop1[1] in cands1:
        #     total_score += 1
        #     total_score_hop1 += 1

        cands2 = [item[1] for item in cand_hop2]
        for cand in cands2:
            score = fuzz.ratio(supp_hop2[1], cand)
            if score >= 80:
                total_score += 1
                total_score_hop2 += 1 
                break       
        # if supp_hop2[1] in cands2:
        #     total_score += 1
        #     total_score_hop2 += 1

    print('Total: ', total_score, total_score/(2*len(data)))
    print('Hop1: ', total_score_hop1, total_score_hop1/len(data))
    print('Hop2: ', total_score_hop2, total_score_hop2/len(data))


def main():
    model_name = "rsvp-ai/bertserini-bert-base-squad"
    tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    #bert_reader = BERT(model_name, tokenizer_name)
    searcher = build_searcher("/data/mo.169/index/lucene-index.enwiki-20180701-paragraphs")

    question = "What is the mascot of the team that has Nicholas S. Zeppos as its leader? Nicholas S. Zeppos (born 1954) is an American lawyer and university administrator.\t He serves as the eighth Chancellor of Vanderbilt University in Nashville, Tennessee.\t He is one of the highest-paid university presidents in the United States."
    retrieve_cand_paragraph(question,'', searcher)

if __name__ == '__main__':
    main()