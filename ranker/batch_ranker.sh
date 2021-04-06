

CUDA_VISIBLE_DEVICES=0,1 python -u run_hybrid_ranker.py \
--train=True \
--cwq_text_path='../data/demo_cwq_train_text_cand.pkl' \
--cwq_kb_path='../data/demo_cwq_train_kb_cand.pkl' \
--hpqa_text_path='../data/demo_hotpotqa_train_text_cand.pkl' \
--hpqa_kb_path='../data/demo_hotpotqa_train_kb_cand.pkl' \
--output_dir='./saved_models' \
--train_batch_size=2 \
--cwq_text_sample=5 \
--cwq_kb_sample=5 \
--hpqa_text_sample=5 \
--hpqa_kb_sample=5 \
--print_step_interval=100


# --cwq_text_path='../data/cwq_train_cand_paragraphs.pkl' \
# --cwq_kb_path='../data/cwq_train_cand_paths.pkl' \
# --hpqa_text_path='../data/hotpotqa_train_cand_paragraphs_iclr_tfidf_hyperlink_w_aw.pkl' \
# --hpqa_kb_path='../data/hotpotqa_train_cand_paths_with_score.pkl' \


# --cwq_text_path='../data/demo_cwq_train_text_cand.pkl' \
# --cwq_kb_path='../data/demo_cwq_train_kb_cand.pkl' \
# --hpqa_text_path='../data/demo_hotpotqa_train_text_cand.pkl' \
# --hpqa_kb_path='../data/demo_hotpotqa_train_kb_cand.pkl' \


# --output_dir='../saved_models' \
# --output_suffix='hybrid' \
# --lr=3e-5 --epochs=3 \
# --train_batch_size=2 --grad_acc_steps=1 \
# --num_bm_cands=20 --num_hy_cands=10 \
# --num_kb_cands=30 \
# --chunk_size=10 \
# --close_tqdm=True