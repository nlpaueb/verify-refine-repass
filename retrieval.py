import os
import re
import ast
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from spacy.lang.en import English
import voyageai
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Download NLTK data if not already available
nltk.download('punkt')

def clone_and_load_documents(repo_url, documents_path, doc_range):
    """Clone a repository and load JSON documents from a specified path."""
    if not os.path.exists(documents_path):
        os.system(f'git clone {repo_url}')
        print(f'Repository cloned from {repo_url}')
    else:
        print('Repository already exists.')

    docs = []
    for i in range(doc_range[0], doc_range[1] + 1):
        file_path = os.path.join(documents_path, f'{i}.json')
        try:
            with open(file_path, 'r') as file:
                docs.append(json.load(file))
        except FileNotFoundError:
            print(f'Warning: File {file_path} not found.')

    passages = [passage for doc in docs for passage in doc]
    print(f'Loaded {len(docs)} documents containing {len(passages)} passages.')
    return passages

REPO_URL = "https://github.com/RegNLP/ObliQADataset.git"
DOCUMENTS_PATH = "ObliQADataset/StructuredRegulatoryDocuments"

docs = clone_and_load_documents(REPO_URL, DOCUMENTS_PATH, (1, 40))

def enhance(docs: list, model_name: str = 'voyageai/voyage-law-2') -> list:
    """
    Enhances document metadata by adding sentence segmentation, token counts, and character counts.

    Args:
        docs (list): List of documents to process.
        model_name (str): Name of the tokenizer model.

    Returns:
        list: Enhanced documents with metadata.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if model_name != "cl100k_base"
        else tiktoken.get_encoding("cl100k_base")
    )

    for passage in tqdm(docs, desc="Enhancing documents"):
        passage["Combined"] = passage["PassageID"] + " " + passage["Passage"]
        passage["char_count"] = len(passage["Combined"])
        passage["tokens_count"] = (
            len(tokenizer(passage["Combined"], truncation=True)[0])
            if model_name != "cl100k_base"
            else len(tokenizer.encode(passage["Combined"]))
        )
        passage["sentences"] = list(nlp(passage["Combined"]).sents)
        passage["sentences"] = [str(sentence) for sentence in passage["sentences"]]
        passage["sentence_count"] = len(passage["sentences"])

    return docs


def token_based_split(passage: dict, model_name: str, tokenizer, max_tokens: int) -> list:
    """
    Splits a passage into smaller chunks based on a maximum token count.

    Args:
        passage (dict): The passage dictionary containing sentences.
        model_name (str): Name of the tokenizer model.
        tokenizer: Tokenizer object.
        max_tokens (int): Maximum allowed tokens per chunk.

    Returns:
        list: List of sentence chunks.
    """
    sentence_chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for sentence in passage["sentences"]:
        sentence_tokens = (
            len(tokenizer(sentence, truncation=True)[0])
            if model_name != "cl100k_base"
            else len(tokenizer.encode(sentence))
        )

        if current_chunk_tokens + sentence_tokens > max_tokens:
            sentence_chunks.append(current_chunk)
            current_chunk = []
            current_chunk_tokens = 0

        current_chunk.append(sentence)
        current_chunk_tokens += sentence_tokens

    sentence_chunks.append(current_chunk)
    return sentence_chunks


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model: str) -> list[float]:
    """
    Retrieves a single embedding for a given text.

    Args:
        text (str): Text to embed.
        model (str): Model to use for embedding.

    Returns:
        list[float]: Embedding vector.
    """
    return client.embeddings.create(input=[text], model=model).data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input_texts: list[str]) -> list[list[float]]:
    """
    Retrieves embeddings for a batch of texts.

    Args:
        input_texts (list[str]): List of texts to embed.

    Returns:
        list[list[float]]: List of embedding vectors.
    """
    response = client.embeddings.create(input=input_texts, model=EMBEDDING_MODEL).data
    return [data.embedding for data in response]


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def embed_with_backoff(**kwargs):
    """
    Embeds texts using the VoyageAI client with exponential backoff.

    Args:
        **kwargs: Arguments for the VoyageAI embedding function.

    Returns:
        list: Embedding results.
    """
    return vo.embed(**kwargs)


def create_embeddings(
    psg_chunks: list,
    content_key: str,
    emb_model_name: str,
    batch_size: int = 100,
    input_type: str = "document",
) -> list:
    """
    Creates embeddings for a list of text chunks in batches.

    Args:
        psg_chunks (list): List of text chunks to embed.
        content_key (str): Key in each chunk for the text to embed.
        emb_model_name (str): Embedding model name.
        batch_size (int): Batch size for embedding. Defaults to 100.
        input_type (str): Type of input for the embedding. Defaults to "document".

    Returns:
        list: List of embedding vectors.
    """
    embeddings_list = []
    for i in tqdm(range(0, len(psg_chunks), batch_size), desc="Creating embeddings"):
        batch_sentences = [item[content_key] for item in psg_chunks[i : i + batch_size]]
        batch_embeddings = (
            get_embeddings(batch_sentences)
            if emb_model_name == "text-embedding-3-large"
            else embed_with_backoff(
                texts=batch_sentences, model=emb_model_name, input_type=input_type
            ).embeddings
        )
        embeddings_list.extend(batch_embeddings)
    return embeddings_list

# -----------------------

# Load OpenAI and VoyageAI credentials
open_api_key = ""  # Your OpenAI API key
voyage_api_key = ""  # Your VoyageAI API key

client = OpenAI(api_key=open_api_key)
vo = voyageai.Client(api_key=voyage_api_key)

GPT_MODEL = "gpt-4o-2024-08-06"
EMBEDDING_MODEL = "text-embedding-3-large"

model_name = "voyageai/voyage-law-2"
docs = enhance(docs, model_name)

# Split passages into sentence chunks
psg_chunks = []
tokenizer = AutoTokenizer.from_pretrained(model_name) if model_name != "cl100k_base" else tiktoken.get_encoding("cl100k_base")
for item in tqdm(docs, desc="Splitting into sentence chunks"):
    item["sentence_chunks"] = token_based_split(
        item, model_name, tokenizer, max_tokens=16000
    )
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict['ID'] = item['ID']
        chunk_dict['DocumentID'] = item["DocumentID"]
        chunk_dict['PassageID'] = item['PassageID']

        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" => ". A" (will work for any captial letter)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get some stats on our chunks
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(tokenizer(chunk_dict["sentence_chunk"], truncation=True)[0]) if model_name != "cl100k_base" else len(tokenizer.encode(chunk_dict["sentence_chunk"]))

        psg_chunks.append(chunk_dict)


# Generate embeddings for passages
embeddings_list = create_embeddings(psg_chunks, "sentence_chunk", "voyage-law-2", 128)
for i in range(len(psg_chunks)):
    psg_chunks[i]["embedding"] = embeddings_list[i]

# Load test set questions
with open("ObliQADataset/ObliQA_test.json") as f:
    questions = json.load(f)

# Generate embeddings for questions
question_embeddings_list = create_embeddings(
    questions, "Question", "voyage-law-2", 128, input_type="query"
)
for i in range(len(questions)):
    questions[i]["embedding"] = question_embeddings_list[i]

# Save embeddings to CSV
if not os.path.exists("embeddings"):
    print("Directory 'embeddings' does not exist. Creating it...")
    os.makedirs("embeddings")
else:
    print("Directory 'embeddings' already exists.")

embeddings_name = "vl2"
psg_chunks_embeddings_df = pd.DataFrame(psg_chunks)
psg_chunks_embeddings_df.to_csv(
    f"embeddings/{embeddings_name}_embeddings_df.csv", index=False
)
question_embeddings_df = pd.DataFrame(questions)
question_embeddings_df.to_csv(
    f"embeddings/{embeddings_name}_question_embeddings_df.csv", index=False
)

# Repeat for voyage-finance-2
docs = clone_and_load_documents(REPO_URL, DOCUMENTS_PATH, (1, 40))
model_name = "voyageai/voyage-finance-2"
docs = enhance(docs, model_name)

# Split passages into sentence chunks
psg_chunks = []
tokenizer = AutoTokenizer.from_pretrained(model_name) if model_name != "cl100k_base" else tiktoken.get_encoding("cl100k_base")
for item in tqdm(docs, desc="Splitting into sentence chunks"):
    item["sentence_chunks"] = token_based_split(
        item, model_name, tokenizer, max_tokens=16000
    )
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict['ID'] = item['ID']
        chunk_dict['DocumentID'] = item["DocumentID"]
        chunk_dict['PassageID'] = item['PassageID']

        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" => ". A" (will work for any captial letter)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get some stats on our chunks
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(tokenizer(chunk_dict["sentence_chunk"], truncation=True)[0]) if model_name != "cl100k_base" else len(tokenizer.encode(chunk_dict["sentence_chunk"]))

        psg_chunks.append(chunk_dict)


# Generate embeddings for passages
embeddings_list = create_embeddings(psg_chunks, "sentence_chunk", "voyage-finance-2", 128)
for i in range(len(psg_chunks)):
    psg_chunks[i]["embedding"] = embeddings_list[i]

# Load test set questions
with open("ObliQADataset/ObliQA_test.json") as f:
    questions = json.load(f)

# Generate embeddings for questions
question_embeddings_list = create_embeddings(
    questions, "Question", "voyage-finance-2", 128, input_type="query"
)
for i in range(len(questions)):
    questions[i]["embedding"] = question_embeddings_list[i]

# Save embeddings to CSV
if not os.path.exists("embeddings"):
    print("Directory 'embeddings' does not exist. Creating it...")
    os.makedirs("embeddings")
else:
    print("Directory 'embeddings' already exists.")

embeddings_name = "vf2"
psg_chunks_embeddings_df = pd.DataFrame(psg_chunks)
psg_chunks_embeddings_df.to_csv(
    f"embeddings/{embeddings_name}_embeddings_df.csv", index=False
)
question_embeddings_df = pd.DataFrame(questions)
question_embeddings_df.to_csv(
    f"embeddings/{embeddings_name}_question_embeddings_df.csv", index=False
)

print("Saved all embeddings")
#--------------------------

def ensure_directory_exists(directory_name: str):
    """
    Ensures a directory exists. Creates it if it does not exist.
    """
    if not os.path.exists(directory_name):
        print(f"Directory '{directory_name}' does not exist. Creating it...")
        os.makedirs(directory_name)
    else:
        print(f"Directory '{directory_name}' already exists.")

# Retrieve relevant passages
def retrieve_relevant_passages(query_embedding, embeddings_tensor, top_k):
    """
    Retrieves top-k passages based on similarity (dot product) with query embedding.
    """
    dot_scores = embeddings_tensor @ query_embedding
    return torch.topk(dot_scores, top_k)

# Load embeddings
def load_embeddings(folder_path: str) -> dict:
    """
    Loads all .csv files from a specified folder containing embeddings into a dictionary.
    """
    embeddings_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            # Load embeddings DataFrame
            df_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(df_path)
            df['embedding'] = df['embedding'].apply(ast.literal_eval)

            embeddings_list = df['embedding'].tolist()
            embeddings_tensor = torch.tensor(np.array(embeddings_list))

            # Classify as passage or query embeddings
            if "question" in file_name.lower():
                base_name = file_name.replace("_question_embeddings_df.csv", "")
                embeddings_data.setdefault(base_name, {})
                embeddings_data[base_name]['test_set'] = df.to_dict(orient='records')
                embeddings_data[base_name]['query_embeddings_tensor'] = embeddings_tensor
            else:
                base_name = file_name.replace("_embeddings_df.csv", "")
                embeddings_data.setdefault(base_name, {})
                embeddings_data[base_name]['passage_chunks'] = df.to_dict(orient='records')
                embeddings_data[base_name]['embeddings_tensor'] = embeddings_tensor
    return embeddings_data

def normalize_scores(scores: np.array) -> np.array:
    """
    Normalize an array of scores to the range [0, 1].

    Args:
        scores (np.array): Array of scores to be normalized.

    Returns:
        np.array: Normalized scores.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    divisor = max(max_score - min_score, 1e-4)
    return (scores - min_score) / (divisor)

def rank_fusion(scores_x: np.array, scores_y: np.array, a: float = 0.5, top_k: int = 10) -> tuple:
    """
    Perform rank fusion by combining two sets of scores.

    Args:
        scores_x (np.array): First set of scores.
        scores_y (np.array): Second set of scores.
        a (float, optional): Weight for `scores_x` in the fusion. Defaults to 0.5.
        top_k (int, optional): Number of top results to return. Defaults to 10.

    Returns:
        tuple: 
            - List of top fused scores.
            - List of indices corresponding to the top scores.
    """
    norm_x = normalize_scores(scores_x)
    norm_y = normalize_scores(scores_y)

    fusion_scores = a * norm_x + (1 - a) * norm_y    

    top_indices = np.argsort(fusion_scores)[::-1][:top_k].tolist()
    top_scores = fusion_scores[top_indices].tolist()

    return top_scores, top_indices

def rank_fusion_on_three(scores_x: np.array, scores_y: np.array, scores_z: np.array, a: float = 0.5, b: float = 0.25, top_k: int = 10) -> tuple:
    """
    Perform rank fusion by combining three sets of scores.

    Args:
        scores_x (np.array): First set of scores.
        scores_y (np.array): Second set of scores.
        scores_z (np.array): Third set of scores.
        a (float, optional): Weight for `scores_x` in the fusion. Defaults to 0.5.
        b (float, optional): Weight for `scores_y` in the fusion. Defaults to 0.25.
        top_k (int, optional): Number of top results to return. Defaults to 10.

    Returns:
        tuple: 
            - List of top fused scores.
            - List of indices corresponding to the top scores.
    """
    norm_x = normalize_scores(scores_x)
    norm_y = normalize_scores(scores_y)
    norm_z = normalize_scores(scores_z)

    fusion_scores = a * norm_x + b * norm_y + (1 - (a + b)) * norm_z

    top_indices = np.argsort(fusion_scores)[::-1][:top_k].tolist()
    top_scores = fusion_scores[top_indices].tolist()

    return top_scores, top_indices

# Simple retrieval
def simple_retrieval(
        query: dict, 
        embeddings_tensor: torch.tensor, 
        top_k: int, 
        passages_list: list) -> list:
    """
    Retrieves top-k unique passages for a query, ensuring no duplicate document IDs.
    """
    extra_passages = embeddings_tensor.shape[0] - 13732  # Handle extra passages
    query_emb = torch.tensor(np.array(query["embedding"]))
    top_scores, top_indices = retrieve_relevant_passages(query_emb, embeddings_tensor, top_k + extra_passages)

    unique_results = []
    seen_docs = set()
    for i in range(top_k + extra_passages):
        doc_id = passages_list[top_indices[i]]['ID']
        if doc_id not in seen_docs:
            unique_results.append((doc_id, top_scores[i]))
            seen_docs.add(doc_id)

        if len(unique_results) == top_k:  # Stop once top-k unique results are found
            break
    return unique_results

# Rank fusion for two retrievers
def rank_fusion_retrieval(
        query: tuple, 
        retriever_1, 
        retriever_2, 
        passages_list: list, 
        top_k: int, a: 
        float = 0.5) -> list:
    """
    Combines results from two retrievers (BM25 or neural embeddings) using rank fusion.
    """
    extra_passages = 0

    # Handle retriever_1 scores
    if hasattr(retriever_1, "get_scores"):  # BM25 retriever
        retriever1_scores = retriever_1.get_scores(query[0]["tokenized_text"])
    elif isinstance(retriever_1, torch.Tensor):  # Neural retriever as torch.Tensor object
        extra_passages = retriever_1.shape[0] - 13732
        query_emb = torch.tensor(np.array(query[0]["embedding"]))
        retriever1_scores = np.array(retriever_1 @ query_emb)
    else:
        raise ValueError("Unsupported retriever type for retriever_1")

    # Handle retriever_2 scores
    if hasattr(retriever_2, "get_scores"):  # BM25 retriever
        retriever2_scores = retriever_2.get_scores(query[1]["tokenized_text"])
    elif isinstance(retriever_2, torch.Tensor):  # Neural retriever as torch.Tensor object
        extra_passages = retriever_2.shape[0] - 13732
        query_emb = torch.tensor(np.array(query[1]["embedding"]))
        retriever2_scores = np.array(retriever_2 @ query_emb)
    else:
        raise ValueError("Unsupported retriever type for retriever_2")

    # Rank fusion calculation
    top_scores, top_indices = rank_fusion(retriever1_scores, retriever2_scores, a, top_k + extra_passages)

    # Extract top-k unique results
    unique_results = []
    seen_docs = set()
    for i in range(top_k + extra_passages):
        doc_id = passages_list[top_indices[i]]['ID']
        if doc_id not in seen_docs:
            unique_results.append((doc_id, top_scores[i]))
            seen_docs.add(doc_id)
        if len(unique_results) == top_k:  # Stop once top-k results are found
            break
    return unique_results

# Triple rank fusion
def triple_rank_fusion_retrieval(
        query: tuple, 
        retriever_1, 
        retriever_2, 
        retriever_3, 
        passages_list: list, 
        top_k: int, 
        a: float = 0.33, 
        b: float = 0.33, 
        reranking: bool = False, 
        top_k_init: int = 10, 
        vo: object = None) -> list:
    """
    Combines results from three retrievers (BM25 or neural embeddings) using weighted rank fusion.
    """
    def get_scores(retriever, query):
        if hasattr(retriever, "get_scores"):
            return retriever.get_scores(query["tokenized_text"]), 0
        elif isinstance(retriever, torch.Tensor):
            extra_passages = retriever.shape[0] - 13732
            query_emb = torch.tensor(np.array(query["embedding"]))
            return np.array(retriever @ query_emb), extra_passages
        else:
            raise ValueError("Unsupported retriever type")

    scores_1, extra_passages = get_scores(retriever_1, query[0])
    scores_2, extra_passages = get_scores(retriever_2, query[1])
    scores_3, extra_passages = get_scores(retriever_3, query[2])

    top_scores, top_indices = rank_fusion_on_three(scores_1, scores_2, scores_3, a, b, top_k + extra_passages)

    if reranking:
        unique_indices = []
        seen_docs = set()
        for j in range(top_k_init + extra_passages):
            doc_id = passages_list[top_indices[j]]['ID']
            if doc_id not in seen_docs:
                unique_indices.append(top_indices[j])
                seen_docs.add(doc_id)

            # Stop once we have the top_k unique results
            if len(unique_indices) == top_k_init:
                break
        
        # Then we use the voyage rerank-2
        question = query[0]['Question']
        passages = [passages_list[k]['sentence_chunk'] for k in unique_indices]

        reranking = vo.rerank(question, passages, model="rerank-2", top_k=top_k)

        results = []
        for r in reranking.results:
            k = unique_indices[r.index]
            results.append((passages_list[k]['ID'], r.relevance_score))

        return results
    else:
        # Extract top-k unique results
        unique_results = []
        seen_docs = set()
        for i in range(top_k + extra_passages):
            doc_id = passages_list[top_indices[i]]['ID']
            if doc_id not in seen_docs:
                unique_results.append((doc_id, top_scores[i]))
                seen_docs.add(doc_id)

            # Stop once we have the top_k unique results
            if len(unique_results) == top_k:
                break

        return unique_results

# Save rankings to TREC file format
def save_rankings_to_trec_file(retrieved: dict, file_path: str, run_name: str):
    """
    Saves the retrieved rankings to a .trec file in the required format.
    """
    with open(file_path, "w") as f:
        for qid, hits in retrieved.items():
            for i, (docid, score) in enumerate(hits):
                line = f"{qid} 0 {docid} {i+1} {score} {run_name}"
                f.write(line + "\n")

#-----------------------------------

ensure_directory_exists("retrieval")
folder_path = "embeddings"
embeddings_dict = load_embeddings(folder_path)

# Example of simple retrieval
retriever = 'vl2'
top_k = 10
retrieved = {}
for query in tqdm(embeddings_dict[retriever]['test_set'], desc="Neural retrieval with voyage-law-2"):
    retrieved[query["QuestionID"]] = simple_retrieval(
        query, 
        embeddings_dict[retriever]['embeddings_tensor'], 
        top_k=10, 
        passages_list=embeddings_dict[retriever]['psg_chunks'])
    
save_rankings_to_trec_file(retrieved, f"retrieval/rankings_{retriever}_{top_k}.trec", retriever
)

# Example of rank-fusion on pairs for BM25 and voyage-law-2
retriever = 'vl2'
tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage-law-2')

for chunk in embeddings_dict[retriever]['psg_chunks']:
    chunk["tokenized_text"] = tokenizer.tokenize(chunk['sentence_chunk'])

for question in embeddings_dict[retriever]['test_set']:
    question["tokenized_text"] = tokenizer.tokenize(question['Question'])

# Extract tokenized chunks and questions into separate lists
tokenized_chunks = [chunk["tokenized_text"] for chunk in embeddings_dict[retriever]['psg_chunks']]
tokenized_questions = [question["tokenized_text"] for question in embeddings_dict[retriever]['test_set']]

bm25 = BM25Okapi(tokenized_chunks)

a = 0.25
top_k = 10
retrieved = {}
for i in tqdm(range(len(embeddings_dict[retriever]['test_set'])), desc="Rank-fusion with BM25 and voyage-law-2"):

    query = embeddings_dict[retriever]['test_set'][i]
    retrieved[embeddings_dict[retriever]['test_set'][i]["QuestionID"]] = rank_fusion_retrieval(
        (query, query), 
        bm25, 
        embeddings_dict[retriever]['embeddings_tensor'], 
        embeddings_dict[retriever]['psg_chunks'], 
        top_k, a
        )

save_rankings_to_trec_file(
    retrieved, 
    f"retrieval/rankings_rf_bm25_{retriever}_{top_k}_{int(a*100)}.trec", 
    f"rf_bm25_{retriever}"
)

# Example of triple rank-fusion without reranking
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

for chunk in embeddings_dict['vl2']['psg_chunks']:
    chunk["tokenized_text"] = word_tokenize(chunk['sentence_chunk'])

for question in embeddings_dict['vl2']['test_set']:
    question["tokenized_text"] = word_tokenize(question['Question'])

# Extract tokenized chunks and questions into separate lists
tokenized_chunks = [chunk["tokenized_text"] for chunk in embeddings_dict['vl2']['psg_chunks']]
tokenized_questions = [question["tokenized_text"] for question in embeddings_dict['vl2']['test_set']]

bm25 = BM25Okapi(tokenized_chunks)

retriever1 = 'vl2'
retriever2 = 'vf2'

a = 0.25
b = 0.20
top_k = 10
retrieved = {}
for i in tqdm(range(len(embeddings_dict[retriever1]['test_set'])), desc="Triple rank-fusion"):
    
    query1 = embeddings_dict[retriever1]['test_set'][i]
    query2 = embeddings_dict[retriever2]['test_set'][i]
    retrieved[embeddings_dict[retriever1]['test_set'][i]["QuestionID"]] = triple_rank_fusion_retrieval(
        (query1, query1, query2), 
        bm25, 
        embeddings_dict[retriever1]['embeddings_tensor'], 
        embeddings_dict[retriever2]['embeddings_tensor'], 
        embeddings_dict[retriever1]['psg_chunks'], 
        top_k=top_k, a=a, b=b
        )

save_rankings_to_trec_file(
    retrieved, 
    f"retrieval/rankings_trf_{top_k}_a{int(a*100)}_b{int(b*100)}.trec", 
    "triple_rank_fusion_bm25_law_finance"
    )

# Example of triple rank-fusion with reranking
bm25 = BM25Okapi(tokenized_chunks)

retriever1 = 'vl2'
retriever2 = 'vf2'

voyage_api_key = '' # your VoyageAI api key
vo = voyageai.Client(api_key=voyage_api_key)

a = 0.25
b = 0.20
top_k = 10
top_k_init = 50
retrieved = {}
for i in tqdm(range(len(embeddings_dict[retriever1]['test_set'])), desc="Triple rank-fusion with reranking"):

    query1 = embeddings_dict[retriever1]['test_set'][i]
    query2 = embeddings_dict[retriever2]['test_set'][i]
    retrieved[embeddings_dict[retriever1]['test_set'][i]["QuestionID"]] = triple_rank_fusion_retrieval(
        (query1, query1, query2), 
        bm25, 
        embeddings_dict[retriever1]['embeddings_tensor'], 
        embeddings_dict[retriever2]['embeddings_tensor'], 
        embeddings_dict[retriever1]['psg_chunks'], 
        top_k=top_k, 
        a=a, b=b, reranking=True, top_k_init=top_k_init, vo=vo
        )

save_rankings_to_trec_file(
    retrieved, 
    f"retrieval/rankings_fusion_bm25_vl2_vf2_rerank2_{top_k_init}_{top_k}.trec", 
    "triple_rank_fusion_bm25_voyage_law2_finance2_rerank2"
    )