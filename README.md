## Datasets
https://huggingface.co/datasets/LDI-lab/trec-rag-2024
### format
`bm25_score` is just a key indicating score.
`qrels' filled up if available.
```
{‘qid’: xx, ‘query’: text, ‘top1000’: [{‘docid’: xx, ‘rank’: 1, ‘bm25_score’: xx, ‘text’: xxx}, {...}], ‘qrels’: {‘docidxxx’: 0/1/2, ..}}
```
