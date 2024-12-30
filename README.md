# AUEB-Archimedes at RIRAG-2025: Is obligation concatenation really all you need?

This repository contains Python scripts describing our systems for the RIRAG-2025 shared task. It is designed to support research in RAG (Retrieval-Augmented Generation) systems. This project leverages a combination of statistical and neural retrieval techniques, neural rerankers, and advanced generative models, with a focus on optimizing performance for the RePASs evaluation metric.

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.11 or higher
- Required libraries:
  ```bash
  pip install tqdm
   pip install numpy
   pip install torch
   pip install transformers
   pip install scikit-learn
   pip install nltk
   pip install spacy
   pip install openai
   pip install pandas
   pip install tiktoken
   pip install rank-bm25
   pip install tenacity
   pip install -U voyageai
  ```

## Project Overview

This repository consists of scripts structured to address the subtasks of the RIRAG-2025 shared task:

1. **Passage Retrieval**:
   - Retrieve the top-10 most relevant passages from a regulatory text corpus.
   - Implement advanced techniques such as **Rank Fusion** and **Neural Reranking**.

2. **Answer Generation**:
   - Generate coherent, accurate answers based on retrieved passages.
   - Employ iterative refinement techniques to enhance answer quality by reducing contradictions and increasing coverage of extracted obligations.

## Files Overview

### 1. `retrieval.py`
   - Implements passage retrieval pipelines using:
     - **BM25** 
     - **Neural embedding-based retrieval** with models like `voyage-law-2` and `voyage-finance-2`.
   - Includes functions for:
     - Rank fusion.
     - Triple-rank fusion with reranking.
   - Outputs TREC-format ranking files.

### 2. `generation.py`
   - Implements passage-based answer generation using LLMs (e.g., GPT-4 and LegalBERT) for question answering.
   - Includes:
     - Iterative refinement of answers.
     - Final scoring and evaluation of answers.

### 3. `prompts.json`
A JSON file containing all the prompts used for our algorithms in the form of a dictionary.

## Running the Experiments

### Passage Retrieval:
1. Run the retrieval pipelines to generate retrieval rankings:
   ```bash
   python retrieval.py
   ```

2. Evaluate the results using metrics such as **recall@10** and **MAP@10**.

### Answer Generation:
1. Process the retrieved passages to generate answers using `generation.py`:
   ```bash
   python generation.py
   ```

2. Evaluate the generated answers using the **RePASs metric**, which includes:
   - Entailment score.
   - Contradiction score.
   - Obligation coverage.

## Notes

1. If you do not have a GPU, ensure to modify the scripts to disable GPU-based operations by setting `device='cpu'`.
2.  Update the paths in `retrieval.py` and `generation.py` to match your local setup, if needed.
3. Certain models may require a Hugging Face account.

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