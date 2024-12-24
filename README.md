# AUEB-Archimedes at RIRAG-2025: Is obligation concatenation really all you need?

This repository contains Jupyter Notebooks describing our systems for the RIRAG-2025 shared task. It is designed to support research in RAG (Retrieval-Augmented Generation) systems. This project leverages a combination of statistical and neural retrieval techniques, neural rerankers, and advanced generative models, with a focus on optimizing performance for the RePASs evaluation metric.

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.11 or higher
- Required libraries:
  ```bash
  pip install tqdm
  pip install numpy
  pip install torch
  pip install spacy
  pip install openai
  pip install pandas
  pip install tiktoken
  pip install rank-bm25
  pip install -U voyageai

  ```
- Jupyter Notebook for running `.ipynb` files (optional).

## Project Overview

This repository consists of utilities, notebooks, and scripts structured to address the subtasks of the RIRAG-2025 shared task:

1. **Passage Retrieval**:
   - Identify the top-10 most relevant passages from a regulatory text corpus.
   - Implement advanced ranking techniques such as Rank Fusion and neural reranking.

2. **Answer Generation**:
   - Generate coherent, accurate answers based on retrieved passages.
   - Employ iterative refinement techniques to enhance answer quality by reducing contradictions and increasing coverage of extracted obligations.

## Files Overview

### 1. `rag_utils.py`
This Python module contains utility functions for handling retrieval and ranking tasks:
- **TREC Parsing**: Parse TREC-formatted files for retrieval results.
- **Score Normalization**: Normalize passage scores for consistency.
- **Rank Fusion**: Combine rankings from multiple retrievers using weighted fusion.

### 2. `retrieval.ipynb`
A Jupyter Notebook for executing retrieval experiments:
- Set up and evaluate retrieval pipelines.
- Test ranking and filtering logic using utilities from `rag_utils.py`.
- Visualize the results.

### 3. `generation.ipynb`
A Jupyter Notebook for running passage generation experiments:
- Passage-based answer generation workflows.
- Generation using LLMs and iterative refinement to improve answer coherence and obligation coverage.

### 4. `prompts.json`
A JSON file containing all the prompts used for our algorithms in the form of a dictionary.

## Running the Experiments
You can replicate our experiments by running the two Jupyter Notebooks.

### Using the Rank Fusion Functionality
1. Import `rag_utils.py` into your Python script:
   ```python
   from rag_utils import *
   ```
2. Provide the scores from multiple retrievers and get the fused rankings:
   ```python
   fused_scores, fused_indices = rank_fusion(scores_x, scores_y, a=0.5, top_k=10)
   ```

## Notes

1. If you do not have a GPU, ensure to modify the scripts to disable GPU-based operations by setting `device='cpu'`.
2. For replicating experiments in the provided notebooks, replace placeholder paths with your dataset paths.
3. The `generation.ipynb` notebook relies on Hugging Face models. Make sure to log in to your Hugging Face account if required.
4. For evaluating each subtask and experiment:
   - For passage retrieval, metrics like `recall@10` and `MAP@10` are used.
   - For answer generation, the  **RePASs metric** is utilized to assess entailment, contradiction, and obligation coverage.

### BibTeX

```shell
@misc{chasandras2024auebarchimedesrirag2025obligationconcatenation,
      title={AUEB-Archimedes at RIRAG-2025: Is obligation concatenation really all you need?}, 
      author={Ioannis Chasandras and Odysseas S. Chlapanis and Ion Androutsopoulos},
      year={2024},
      eprint={2412.11567},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.11567}, 
}