# SimultQA


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
