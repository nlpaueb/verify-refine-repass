import numpy as np
import json

def parse_trec_file(file_path: str):
    """
    Recreates the retrieved dictionary that represents the results of the retrieval stage
    """
    trec_dict = {}
    
    # open and read the TREC file line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  
            question_id = parts[0]  
            passage_id = parts[2]  
            score = float(parts[4]) 

            # if the QuestionID is not in the dictionary, add it
            if question_id not in trec_dict:
                trec_dict[question_id] = []
                
            trec_dict[question_id].append((passage_id, score))
    
    return trec_dict

def normalize_scores(scores:np.array):
    min_score = np.min(scores)
    max_score = np.max(scores)
    divisor = max(max_score - min_score, 1e-4)
    return (scores - min_score) / (divisor)

def rank_fusion(scores_x:np.array, scores_y:np.array, a=0.5, top_k=10):
    norm_x = normalize_scores(scores_x)
    norm_y = normalize_scores(scores_y)

    fusion_scores = a*norm_x + (1-a)*norm_y    

    top_indices = np.argsort(fusion_scores)[::-1][:top_k].tolist()
    top_scores = fusion_scores[top_indices].tolist()

    return top_scores, top_indices

def rank_fusion_on_three(scores_x:np.array, scores_y:np.array, scores_z:np.array, a=0.5, b=0.25, top_k=10):
    norm_x = normalize_scores(scores_x)
    norm_y = normalize_scores(scores_y)
    norm_z = normalize_scores(scores_z)

    fusion_scores = a*norm_x + b*norm_y + (1-(a+b))*norm_z

    top_indices = np.argsort(fusion_scores)[::-1][:top_k].tolist()
    top_scores = fusion_scores[top_indices].tolist()

    return top_scores, top_indices

def keep_qualifying(psg_tuples, threshold=0.7, max_drop=0.2, keep_one=True):
    first = psg_tuples[0]
    # keep only the passages with high relevance
    for i in range(len(psg_tuples)):
        if psg_tuples[i][1] < threshold :
            psg_tuples = psg_tuples[:i]
            break

    # keep only the passages for which no significant drop was observed
    for i in range(1, len(psg_tuples)):
        if psg_tuples[i-1][1] - psg_tuples[i][1] >= max_drop:
            psg_tuples = psg_tuples[:i]
            break

    # keep at least one answer if none satisfy the criteria
    if not psg_tuples and keep_one: psg_tuples.append(first)

    return psg_tuples