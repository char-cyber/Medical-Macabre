import pandas as pd
import numpy as np
# New file: active_learning.py
def active_learning_loop(
    model, 
    unlabeled_pool: pd.DataFrame, 
    n_rounds: int = 3,
    n_query: int = 500
) -> pd.DataFrame:
    """
    Uncertainty sampling to find useful sentences from unlabeled data
    """
    labeled_data = []
    
    for round_num in range(n_rounds):
        # Get predictions & uncertainty
        probs = model.predict_proba(unlabeled_pool['text'].tolist())
        uncertainty = 1 - np.max(probs, axis=1)  # Lower confidence = higher uncertainty
        
        # Select most uncertain samples
        query_idx = np.argsort(uncertainty)[-n_query:]
        queries = unlabeled_pool.iloc[query_idx]
        
        # Simulate human labeling (in practice, you'd export for manual review)
        # Here we use heuristics + confidence threshold
        for idx, row in queries.iterrows():
            # High uncertainty + contains ICD keywords = likely useful
            if uncertainty[idx] > 0.3 and contains_icd_keywords(row['text']):
                labeled_data.append({
                    'text': row['text'],
                    'label': 1,
                    'label_binary': 1
                })
            else:
                labeled_data.append({
                    'text': row['text'],
                    'label': 0,
                    'label_binary': 0
                })
        
        # Remove queried samples from pool
        unlabeled_pool = unlabeled_pool.drop(query_idx)
        
        print(f"Round {round_num+1}: Added {len(query_idx)} labeled samples")
    
    return pd.DataFrame(labeled_data)

def contains_icd_keywords(text: str) -> bool:
    keywords = ['diagnos', 'present', 'history', 'admits', 'complaint', 
                'finding', 'result', 'procedure', 'surgery', 'medication']
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)