import os
import json
import requests

# Cloning and loading documents
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

# Loading questions from JSON file
def load_questions(questions_path):
    """Load questions from a specified JSON file."""
    try:
        with open(questions_path, 'r') as file:
            questions = json.load(file)
        print(f'Loaded {len(questions)} questions.')
        return questions
    except FileNotFoundError:
        print(f'Error: Questions file {questions_path} not found.')
        return []
    
def load_json_dataset(file_url, file_path):
    if not os.path.exists(file_path):
        print("File doesn't exist, downloading...")

        # Send a GET request to download the file
        response = requests.get(file_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the content to a local file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("File downloaded successfully!")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    else:
        print("File already exists.")

# Generating lookup dictionaries
def generate_lookup_dicts(passages, questions):
    """Generate dictionaries for efficient lookup of passages and questions."""
    psgid2index = {}
    pid2passage = {}

    for i, psg in enumerate(passages):
        try:
            psg_id = psg['ID']
            if psg_id not in psgid2index:
                psgid2index[psg_id] = i
            if psg_id not in pid2passage:
                pid2passage[psg_id] = psg['Passage']
        except KeyError:
            print(f"Warning: Missing 'ID' or 'Passage' in passage: {psg}")

    qid2question = {}
    for q in questions:
        try:
            qid2question[q['QuestionID']] = q['Question']
        except KeyError:
            print(f"Warning: Missing 'QuestionID' or 'Question' in question: {q}")

    print(f"Generated lookup dictionaries: {len(psgid2index)} passages, {len(qid2question)} questions.")
    return psgid2index, pid2passage, qid2question

# Setting up a directory for storing results
def setup_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Parameters
REPO_URL = "https://github.com/RegNLP/ObliQADataset.git"
DOCUMENTS_PATH = "ObliQADataset/StructuredRegulatoryDocuments"
QUESTIONS_PATH = "ObliQADataset/RIRAGSharedTask/RIRAG_Unseen_Questions.json"
DATASET_URL = "https://raw.githubusercontent.com/RegNLP/ObligationClassifier/main/ObligationClassificationDataset.json"
DOC_RANGE = (1, 40)
OUTPUT_DIR = "generation"
folder_path = ""

# Basic initialization
passages = clone_and_load_documents(REPO_URL, DOCUMENTS_PATH, DOC_RANGE)
questions = load_questions(QUESTIONS_PATH)
load_json_dataset(DATASET_URL, "ObligationClassificationDataset.json")
psgid2index, pid2passage, qid2question = generate_lookup_dicts(passages, questions)
setup_directory(OUTPUT_DIR)

#---------------------------------------------------

import csv
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipeline,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import sent_tokenize as sent_tokenize_uncached
import nltk
from functools import cache
from tqdm.auto import tqdm

# Set up random seeds and deterministic flags for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.environ["WANDB_DISABLED"] = "true"

# Step 2: Load and preprocess the data
json_path = os.path.join(folder_path, "ObligationClassificationDataset.json")
with open(json_path, 'r') as file:
    data = json.load(file)

texts = [item['Text'] for item in data]
labels = [1 if item['Obligation'] else 0 for item in data]  # Converting True/False to 1/0

# Step 3: Tokenization using LegalBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

class ObligationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = ObligationDataset(X_train, y_train, tokenizer)
val_dataset = ObligationDataset(X_val, y_val, tokenizer)

# Step 4: Fine-tuning LegalBERT for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    'nlpaueb/legal-bert-base-uncased', num_labels=2
)
model.to(device)  # Move model to the GPU

# Ensure the directories exist for saving results and logs
output_dir = os.path.join(folder_path, 'results')
log_dir = os.path.join(folder_path, 'logs')
save_dir = os.path.join(folder_path, 'obligation-classifier-legalbert')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,  # Set seed in TrainingArguments
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Step 5: Train the model
trainer.train()

# Step 6: Evaluate the model
trainer.evaluate()

# Step 7: Save the model and tokenizer for future use
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Model fine-tuning and evaluation completed.")

nltk.download('punkt')
nltk.download('punkt_tab')

# Load the tokenizer and model for obligation detection
model_name = os.path.join(folder_path, 'obligation-classifier-legalbert')
obligation_tokenizer = AutoTokenizer.from_pretrained(model_name)
obligation_model = AutoModelForSequenceClassification.from_pretrained(model_name)
obligation_model.to(device)
obligation_model.eval()

# Load NLI model and tokenizer for obligation coverage
coverage_nli_model = pipeline(
    "text-classification", model="microsoft/deberta-large-mnli", device=0 if torch.cuda.is_available() else -1
)

# Load NLI model and tokenizer for entailment and contradiction checks
nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
nli_model.to(device)
nli_model.eval()

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

# Define a cached version of sentence tokenization
@cache
def sent_tokenize(passage: str):
    return sent_tokenize_uncached(passage)

def softmax(logits):
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)

def get_nli_probabilities(premises, hypotheses):
    features = nli_tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    nli_model.eval()
    with torch.no_grad():
        logits = nli_model(**features).logits.cpu().numpy()
    probabilities = softmax(logits)
    return probabilities

def get_nli_matrix(passages, answers):
    entailment_matrix = np.zeros((len(passages), len(answers)))
    contradiction_matrix = np.zeros((len(passages), len(answers)))

    batch_size = 16
    for i, pas in enumerate(passages):
        for b in range(0, len(answers), batch_size):
            e = b + batch_size
            probs = get_nli_probabilities(
                [pas] * len(answers[b:e]), answers[b:e]
            )  # Get NLI probabilities
            entailment_matrix[i, b:e] = probs[:, 1]
            contradiction_matrix[i, b:e] = probs[:, 0]
    return entailment_matrix, contradiction_matrix

def calculate_scores_from_matrix(nli_matrix, score_type='entailment'):
    if nli_matrix.size == 0:
        return 0.0  # or some other default score or handling as appropriate for your use case

    if score_type == 'entailment':
        reduced_vector = np.max(nli_matrix, axis=0)
    elif score_type == 'contradiction':
        reduced_vector = np.max(nli_matrix, axis=0)
    score = np.round(np.mean(reduced_vector), 5)
    return score

def classify_obligations(sentences):
    inputs = obligation_tokenizer(
        sentences, padding=True, truncation=True, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        logits = obligation_model(**inputs).logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return predictions

def calculate_obligation_coverage_score(passages, answers):
    # Filter obligation sentences from passages
    obligation_sentences_source = []
    for passage in passages:
        sentences = sent_tokenize(passage)
        is_obligation = classify_obligations(sentences)
        obligation_sentences_source.extend(
            [sent for sent, label in zip(sentences, is_obligation) if label == 1]
        )

    # Filter obligation sentences from answers
    obligation_sentences_answer = []
    for answer in answers:
        sentences = sent_tokenize(answer)
        is_obligation = classify_obligations(sentences)
        obligation_sentences_answer.extend(
            [sent for sent, label in zip(sentences, is_obligation) if label == 1]
        )

    # Calculate coverage based on NLI entailment
    covered_count = 0
    for obligation in obligation_sentences_source:
        for answer_sentence in obligation_sentences_answer:
            nli_result = coverage_nli_model(
                f"{answer_sentence} [SEP] {obligation}"
            )
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                covered_count += 1
                break

    return (
        covered_count / len(obligation_sentences_source)
        if obligation_sentences_source
        else 0
    )

def calculate_final_composite_score(passages, answers):
    passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
    answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]

    # Calculate NLI matrix for entailment and contradiction
    entailment_matrix, contradiction_matrix = get_nli_matrix(
        passage_sentences, answer_sentences
    )

    # Calculate scores
    entailment_score = calculate_scores_from_matrix(entailment_matrix, 'entailment')
    contradiction_score = calculate_scores_from_matrix(
        contradiction_matrix, 'contradiction'
    )
    obligation_coverage_score = calculate_obligation_coverage_score(passages, answers)

    # Final composite score formula
    composite_score = (
        obligation_coverage_score + entailment_score - contradiction_score + 1
    ) / 3

    # Return all scores
    return (
        np.round(composite_score, 5),
        entailment_score,
        contradiction_score,
        obligation_coverage_score,
    )

def calculate_average_scores_from_csv(output_file_csv):
    """Calculate average scores from the CSV file."""
    entailment_scores = []
    contradiction_scores = []
    obligation_coverage_scores = []
    composite_scores = []

    with open(output_file_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                entailment_scores.append(float(row['entailment_score']))
                contradiction_scores.append(float(row['contradiction_score']))
                obligation_coverage_scores.append(float(row['obligation_coverage_score']))
                composite_scores.append(float(row['composite_score']))
            except ValueError:
                # Handle the case where the value cannot be converted to float
                print(f"Skipping invalid row: {row}")

    avg_entailment = np.mean(entailment_scores) if entailment_scores else 0.0
    avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0.0
    avg_obligation_coverage = (
        np.mean(obligation_coverage_scores) if obligation_coverage_scores else 0.0
    )
    avg_composite = np.mean(composite_scores) if composite_scores else 0.0

    return avg_entailment, avg_contradiction, avg_obligation_coverage, avg_composite

def evaluate(input_file_path, group_method_name):
    # Create a directory with the group_method_name in the folder path
    output_dir = os.path.join(folder_path, group_method_name)
    os.makedirs(output_dir, exist_ok=True)

    # Define the paths for result files
    output_file_csv = os.path.join(output_dir, 'results.csv')
    output_file_txt = os.path.join(output_dir, 'results.txt')

    processed_question_ids = set()
    saved_items_count = 0

    # Check if the output CSV file already exists and read processed QuestionIDs
    if os.path.exists(output_file_csv):
        with open(output_file_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_question_ids.add(row['QuestionID'])
                saved_items_count += 1

    with open(input_file_path, 'r') as file:
        test_data = json.load(file)

    total_items = len(test_data)

    # Open the CSV file for appending results
    with open(output_file_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not processed_question_ids:
            # Write the header if the file is empty or new
            writer.writerow(
                [
                    'QuestionID',
                    'entailment_score',
                    'contradiction_score',
                    'obligation_coverage_score',
                    'composite_score',
                ]
            )

        for index, item in enumerate(test_data, start=1):
            question_id = item['QuestionID']

            # Skip if the QuestionID has already been processed
            if question_id in processed_question_ids:
                continue

            # Skip if the "Answer" is null or empty
            if not item.get('Answer') or not item['Answer'].strip():
                continue

            # Merge "RetrievedPassages" if it's a list
            if isinstance(item['RetrievedPassages'], list):
                item['RetrievedPassages'] = " ".join(item['RetrievedPassages'])

            passages = [item['RetrievedPassages']]
            answers = [item['Answer']]
            (
                composite_score,
                entailment_score,
                contradiction_score,
                obligation_coverage_score,
            ) = calculate_final_composite_score(passages, answers)

            # Write the result to the CSV file
            writer.writerow(
                [
                    question_id,
                    entailment_score,
                    contradiction_score,
                    obligation_coverage_score,
                    composite_score,
                ]
            )

            # Increment the saved items count and print status
            saved_items_count += 1
            print(f"{saved_items_count}/{total_items}")

    # Calculate average scores from the CSV file
    (
        avg_entailment,
        avg_contradiction,
        avg_obligation_coverage,
        avg_composite,
    ) = calculate_average_scores_from_csv(output_file_csv)

    # Print and save results to a text file
    results = (
        f"Average Entailment Score: {avg_entailment}\n"
        f"Average Contradiction Score: {avg_contradiction}\n"
        f"Average Obligation Coverage Score: {avg_obligation_coverage}\n"
        f"Average Final Composite Score: {avg_composite}\n"
    )

    print(results)

    with open(output_file_txt, 'w') as txtfile:
        txtfile.write(results)

    print(f"Processing complete. Results saved to {output_dir}")

#---------------------------------------------------

def parse_trec_file(file_path: str):
    """
    Recreates the retrieved dictionary that represents the results of the retrieval stage.

    Args:
        file_path (str): Path to the TREC file.

    Returns:
        dict: A dictionary mapping question IDs to a list of (passage ID, score) tuples.

    """
    trec_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                question_id, passage_id, score = parts[0], parts[2], float(parts[4])

                if question_id not in trec_dict:
                    trec_dict[question_id] = []

                trec_dict[question_id].append((passage_id, score))

        print(f'Parsed {len(trec_dict)} questions from TREC file.')
    except FileNotFoundError:
        print(f"Error: TREC file {file_path} not found.")
    return trec_dict

def calculate_average(dataset):
    """
    Calculates the average contradiction score for a dataset.

    Args:
        dataset (list): Dataset containing retrieved passages and answers.

    Returns:
        float: Average contradiction score.
    """
    contradiction_scores = []

    for item in tqdm(dataset):
        passages = [" ".join(item['RetrievedPassages'])]
        answers = [item['Answer']]

        passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
        answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]

        # Compute NLI matrices
        entailment_matrix, contradiction_matrix = get_nli_matrix(passage_sentences, answer_sentences)

        # Calculate contradiction score
        contradiction_score = calculate_scores_from_matrix(contradiction_matrix, 'contradiction')
        contradiction_scores.append(contradiction_score)

    return np.mean(contradiction_scores) if contradiction_scores else 0.1

def extract_obligation_sentences(texts):
    """
    Filters and extracts obligation sentences from a list of texts.

    Args:
        texts (list): List of textual inputs (passages or answers).

    Returns:
        list: Sentences classified as obligations.
    """
    obligation_sentences = []
    for text in texts:
        sentences = sent_tokenize(text)
        is_obligation = classify_obligations(sentences)
        obligation_sentences.extend(
            [sent for sent, label in zip(sentences, is_obligation) if label == 1]
        )
    return obligation_sentences

def get_obligation_covering_sentences(passages, answers):
    """
    Finds obligation sentences in answers that cover obligations in passages.
    """
    source_obligations = extract_obligation_sentences(passages)
    answer_obligations = extract_obligation_sentences(answers)

    covering_sentences = []
    for answer_sentence in answer_obligations:
        for obligation in source_obligations:
            nli_result = coverage_nli_model(f"{answer_sentence} [SEP] {obligation}")
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                covering_sentences.append(answer_sentence)
                break
    return covering_sentences    

def get_uncovered_obligations(passages, answers):

    source_obligations = extract_obligation_sentences(passages)
    answer_obligations = extract_obligation_sentences(answers)

    # Check if answers contain any meaningful content
    meaningful_answers = [
        answer.strip() for answer in answers if answer.strip()
    ]  # Remove empty or whitespace-only answers
    if not meaningful_answers:
        return source_obligations

    # Find uncovered obligations based on NLI entailment
    checklist = [False for i in source_obligations]
    for index, obligation in enumerate(source_obligations):
        for answer_sentence in answer_obligations:
            nli_result = coverage_nli_model(
                f"{answer_sentence} [SEP] {obligation}"
            )
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                checklist[index] = True
                break

    return [obl for flag, obl in zip(checklist, source_obligations) if flag==False]

# Helper function for LOC
def is_covered(obligations, answers):
    """
    Determines if any obligations are covered by the answers using NLI entailment.

    Args:
        obligations (list): List of passages or sentences containing obligations.
        answers (list): List of sentences from the answers.

    Returns:
        bool: True if any obligations are covered, otherwise False.
    """

    source_obligations = extract_obligation_sentences(obligations)
    answer_obligations = extract_obligation_sentences(answers)

    # Check for coverage using NLI entailment
    for obligation in source_obligations:
        for answer_sentence in answer_obligations:
            nli_result = coverage_nli_model(
                f"{answer_sentence} [SEP] {obligation}"
            )
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                return True

    return False

def keep_qualifying(psg_tuples, threshold=0.7, max_drop=0.2, keep_one=True):
    """
    Filters passage tuples based on relevance threshold and maximum allowed drop.

    Args:
        psg_tuples (list): List of tuples (passage_id, score).
        threshold (float): Minimum score for a passage to be considered relevant.
        max_drop (float): Maximum allowed drop in scores between consecutive passages.
        keep_one (bool): Whether to keep at least one passage if all are filtered.

    Returns:
        list: Filtered list of passage tuples.
    """
    # Always keep the first passage
    first = psg_tuples[0]

    # Filter passages based on the threshold
    for i in range(len(psg_tuples)):
        if psg_tuples[i][1] < threshold:
            psg_tuples = psg_tuples[:i]
            break

    # Filter passages based on the maximum score drop
    for i in range(1, len(psg_tuples)):
        if psg_tuples[i - 1][1] - psg_tuples[i][1] >= max_drop:
            psg_tuples = psg_tuples[:i]
            break

    # Ensure at least one passage remains
    if not psg_tuples and keep_one:
        psg_tuples.append(first)

    return psg_tuples

def preprocess(retrieval_results: dict, threshold: float = 0.7, max_drop: float = 0.2, keep_obligations: bool = True)->list:

    # Filter passages based on threshold and max_drop
    filtered = {}
    for qid, hit in retrieval_results.items():
        qualified = keep_qualifying(hit, threshold=threshold, max_drop=max_drop, keep_one=True)
        filtered[qid] = qualified

    # Prepare the preprocessed output
    preprocessed = []
    for qid, hit in tqdm(filtered.items()):
        question = qid2question[qid]
        passages = [pid2passage[t[0]] for t in hit]

        # Create the base structure for each question
        preprocessed.append(
            {
            "QuestionID": qid,
            "Question": question,
            "RetrievedPassages": passages,
            }
        )

        # Extract obligation sentences (if enabled)
        if keep_obligations:    

            # For each passage keep only the sentences that are labeled as obligations
            obligations = []
            for passage in passages:
                sentences = sent_tokenize(passage)
                if not sentences: continue
                is_obligation = classify_obligations(sentences)
                obligation_sents = [sent for sent, label in zip(sentences, is_obligation) if label == 1]
                if obligation_sents:
                    obligations.extend(obligation_sents)

            # If no obligations are found, keep the original passages
            if not obligations:
                obligations=passages

            preprocessed[-1]["Obligations"] = obligations

    return preprocessed

from openai import OpenAI

openai_api_key = "" # your OpenAI API key

client = OpenAI(api_key=openai_api_key)
GPT_MODEL = "gpt-4o-2024-08-06" # or "gpt-4o-mini"

with open("prompts.json") as f:
    prompts = json.load(f)

def generate_prompt(input_text: str, context, context_type: str = 'Passages', input_type: str = 'Question')->str:

  if isinstance(context, str):
        context = [context]
  joined_context = "\n".join(context)

  return f"""{context_type}:
  {joined_context}

  {input_type}:
  {input_text}"""

def get_answer(user_prompt: str, system_prompt: str)->str:
    response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": system_prompt
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": user_prompt
          }
        ]
      },
    ],
      temperature=1,
      top_p=1
    )
    return response.choices[0].message.content

def verify(item: dict, N: int, optimization_variable: str) -> dict:
    """
    Verifies and optimizes an answer by generating N possible answers 
    and selecting the best one based on the specified optimization variable.

    Args:
        item (dict): Input dictionary containing question and obligations.
        N (int): Number of answers to generate.
        optimization_variable (str): Variable used to optimize the selected answer 
                                     ('repass', 'entailment', 'contradiction', 'obligations').

    Returns:
        dict: Updated item dictionary with the best answer based on the optimization variable.
    """
    # Generate N answers for the question
    item['Answers'] = []
    for _ in range(N):
        user_prompt = generate_prompt(
            input_text=item['Question'],
            context=item['Obligations'],
            context_type='Obligations'
        )
        answer = get_answer(user_prompt=user_prompt, system_prompt=prompts['Obligations Context Prompt'])
        item['Answers'].append(answer)

    # Evaluate scores for each generated answer
    passages = [" ".join(item['RetrievedPassages'])]
    scores_repass = []
    scores_entailment = []
    scores_contradiction = []
    scores_obligations = []

    for answer in item['Answers']:
        answers = [answer]
        composite_score, entailment_score, contradiction_score, obligation_coverage_score = (
            calculate_final_composite_score(passages, answers)
        )
        scores_repass.append(composite_score)
        scores_entailment.append(entailment_score)
        scores_contradiction.append(contradiction_score)
        scores_obligations.append(obligation_coverage_score)

    # Select the best answer based on the optimization variable
    if optimization_variable == 'repass':
        item['Answer'] = item['Answers'][np.argmax(scores_repass)]

    elif optimization_variable == 'entailment':
        item['Answer'] = item['Answers'][np.argmax(scores_entailment)]

    elif optimization_variable == 'contradiction':
        item['Answer'] = item['Answers'][np.argmin(scores_contradiction)]

    elif optimization_variable == 'obligations':
        item['Answer'] = item['Answers'][np.argmax(scores_obligations)]
    else:
        raise ValueError(f"Invalid optimization variable: {optimization_variable}")

    return item


def refine(item: dict, average: float = 0.02, obl_injection: bool = True) -> dict:
    """
    Refines the selected answer by removing high-contradiction sentences 
    and injecting uncovered obligations if necessary.

    Args:
        item (dict): Input dictionary containing the selected answer, question, and passages.
        average (float): Threshold for identifying high-contradiction sentences.
        obl_injection (bool): Whether to inject uncovered obligations into the answer.

    Returns:
        dict: Updated item dictionary with a refined answer and remaining obligations (if any).
    """
    # Extract sentences from passages and answer
    passages = [" ".join(item['RetrievedPassages'])]
    answers = [item['Answer']]
    passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
    answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]

    # Calculate NLI matrices for entailment and contradiction
    entailment_matrix, contradiction_matrix = get_nli_matrix(passage_sentences, answer_sentences)

    # Identify high-contradiction sentences
    reduced_sents_vector = np.max(contradiction_matrix, axis=0)
    high_contradiction_indices = np.where(reduced_sents_vector > average)[0].tolist()

    # Identify sentences that cover obligations
    covering_sentences = get_obligation_covering_sentences(passages, answers)

    # Remove high-contradiction sentences that do not cover obligations
    new_answer = item['Answer']
    for hci in high_contradiction_indices:
        if answer_sentences[hci] not in covering_sentences:
            new_answer = new_answer.replace(answer_sentences[hci], "")

    item['Answer'] = new_answer.strip()  # Ensure clean formatting

    # Inject uncovered obligations if enabled
    if obl_injection:
        # Identify uncovered obligations
        uncovered_obligations = get_uncovered_obligations(passages, [item['Answer']])
        item['RemainingObligations'] = uncovered_obligations

        # Handle edge cases: No meaningful answers or uncovered obligations
        meaningful_answers = [answer.strip() for answer in answers if answer.strip()]
        if not meaningful_answers and not item['RemainingObligations']:
            item['RemainingObligations'] = item['RetrievedPassages']

        # Inject uncovered obligations into the answer if any remain
        if item['RemainingObligations']:
            user_prompt = generate_prompt(
                input_text=item['Answer'],
                context=item['RemainingObligations'],
                context_type='Obligations',
                input_type='Answer'
            )
            item['Answer'] = get_answer(user_prompt=user_prompt, system_prompt=prompts['Obligation Insertion Prompt'])

    return item


#---------------------------------------------------

retrieval_results = parse_trec_file('rankings.trec')
preprocessed = preprocess(retrieval_results, threshold=0.9, max_drop=0.1)

# VRR
for item in preprocessed:
    item = verify(item=item, N=5, optimization_variable='repass')

ref_it = 3
for i in range(ref_it):
    dataset_avg = calculate_average(preprocessed)
    for item in preprocessed:
        item = refine(item=item, average=dataset_avg)

    if i == ref_it-1:
        item = refine(item=item, average=dataset_avg, obl_injection=False)

output_data = []
for item in preprocessed:
    output_data.append({
        "QuestionID": item['QuestionID'],
        "Question": item['Question'],
        "RetrievedPassages": item['RetrievedPassages'],
        "Answer": item['Answer']
    })

filepath = 'generation'
filename = 'VRR_results.json'
# Write the results to a JSON file
with open(filepath+'/'+filename, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

# NOC
output_data = []
for item in preprocessed:
  output_data.append({
        "QuestionID": item['QuestionID'],
        "Question": item['Question'],
        "RetrievedPassages": item['RetrievedPassages'],
        "Answer": " ".join(item['Obligations']) if item['Obligations'] else " ".join(item['RetrievedPassages'])
    })

filepath = 'generation'
filename = 'NOC_results.json'
with open(filepath+'/'+filename, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

# LOC
for item in tqdm(preprocessed):
  answers_per_obligation = []
  for obligation in item['Obligations']:

    answer = ''
    coverage_flag = False
    tries = 0
    while (not coverage_flag) and (tries < 3):
      # Generate a partial answer
      user_prompt = generate_prompt(input_text=item['Question'], context=obligation, context_type='Obligation')
      answer = get_answer(user_prompt=user_prompt, system_prompt=prompts['LOC Prompt'])
      # Verify that the answer produced covers the obligation
      coverage_flag = is_covered([obligation], [answer])
      tries += 1
    answers_per_obligation.append(answer)
  item['PartialAnswers'] = answers_per_obligation

output_data = []
for item in preprocessed:
  output_data.append({
        "QuestionID": item['QuestionID'],
        "Question": item['Question'],
        "RetrievedPassages": item['RetrievedPassages'],
        "Answer": " ".join(item['PartialAnswers'])
    })

filepath = "generation"
filename = "LOC_results"
with open(filepath+'/'+filename, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

# Example of evaluating results with the NOC algorithm.
group_methodName = 'NOC' # Replace with your desired method name
input_file = os.path.join(folder_path, "generation/NOC_results.json") # Replace with your desired system results
evaluate(input_file, group_methodName)