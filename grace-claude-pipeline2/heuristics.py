

import re

STRONG_KEYWORDS = [
    "pneumonia",
    "sepsis",
    "diabetes",
    "hypertension",
    "heart failure",
    "renal failure",
    "kidney failure",
    "copd",
    "asthma",
    "anemia",
    "fracture",
    "infection",
    "cancer",
    "tumor",
    "embolism",
    "thrombosis",
    "hemorrhage",
    "ischemia",
]


NEGATION_PHRASES = [
    "no evidence of",
    "no signs of",
    "negative for",
    "denies",
    "without evidence of",
    "without signs of",
    "rule out",
    "ruled out",
    "r/o",
    "no",
    "not",
]


HISTORY_PHRASES = [
    "history of",
    "hx of",
    "family history of",
    "prior history of",
]


WINDOW_SIZE = 6


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def has_history_phrase(sentence):
    text = str(sentence).lower()
    return any(phrase in text for phrase in HISTORY_PHRASES)


def keyword_is_negated(tokens, keyword_start_index):
    start = max(0, keyword_start_index - WINDOW_SIZE)
    context = " ".join(tokens[start:keyword_start_index])

    for phrase in NEGATION_PHRASES:
        if phrase in context:
            return True

    return False


def contains_non_negated_keyword(sentence):
    text = str(sentence).lower()
    tokens = tokenize(text)

    for keyword in STRONG_KEYWORDS:
        keyword_tokens = tokenize(keyword)
        keyword_len = len(keyword_tokens)

        for i in range(len(tokens) - keyword_len + 1):
            if tokens[i:i + keyword_len] == keyword_tokens:
                if keyword_is_negated(tokens, i):
                    return False
                if has_history_phrase(sentence):
                    return False
                return True

    return False


def apply_heuristic(sentence, bert_label, probability):
    """
    Only overrides BERT 0 -> 1 when a strong ICD keyword appears
    without negation/history context.
    """

    bert_label = int(bert_label)
    probability = float(probability)

    if bert_label == 0 and contains_non_negated_keyword(sentence):
        return 1, max(probability, 0.99), "keyword_override"

    return bert_label, probability, "bert_prediction"