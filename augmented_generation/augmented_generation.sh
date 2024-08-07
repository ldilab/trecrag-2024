DATASET=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.raggy-dev

python src/ragnarok/scripts/run_ragnarok.py  \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --topk 100,5 \
    --dataset ${DATASET} \
    --retrieval_method bm25,rank_zephyr_rho \
    --prompt_mode chatqa \
    --context_size 8192 \
    --max_output_tokens 1024 