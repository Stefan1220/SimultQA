# SimultQA

## Data Format
```
HotpotQA: []
CWQ: []
```

## Retriever Training
```
CUDA_VISIBLE_DEVICES=0,1 python -u ./retriever/run_retriever.py \
--train=True \
--train_kb_dir='../data/cwq_train_cand_paths.pkl' \
--train_text_dir='../data/cand_paragraphs_iclr_tfidf_hyperlink.pkl' \
--output_dir='../saved_models' \
--output_suffix='hybrid' \
--lr=3e-5 --epochs=3 \
--train_batch_size=2 --grad_acc_steps=1 \
--num_bm_cands=20 --num_hy_cands=10 \
--num_kb_cands=30 \
--chunk_size=10 \
--close_tqdm=True
```

## Retriever Inference
```
python ./retriever/retriever_inference.py --infer_type cwq_kb

# --infer_type can be: cwq_kb, cwq_text, htqa_text, htqa_kb
```


## Evaluation
```
# This script includes three parts: reasoning paths ranking, answer prediction and final evaluation.

python ./eval/prediction_joint.py --dataset_name cwq --kb_beam_size 5 --text_beam_size 5

# --dataset_name can be: cwq, htqa
# --kb_beam_size: the number of kb reasoning paths for ranking
# --text_beam_size: the number of text reasoning paths for ranking

```
