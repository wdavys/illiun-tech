# Find Best Contexts with SQuAD

Retrieve the best (relevant) contexts among a large dataset for a specific query.

## Quick Start Guide

```console
python main.py --help
```

## Note

As transformers are the state-of-the-art architecture yielding to high performance results for NLP-related tasks, we decided to focus on this specific architecture. Hence, we used a ``Sentence Transformer`` called ``all-MiniLM-L6-v2`` to compute the contextual word embeddings of paragraphs in SQuAD.

Provided that loading the model can take some time, embeddings are then cached to allow their use almost instantaneously. The embeddings for the train and dev SQuAD datasets are already provided in ``cache`` folder.
