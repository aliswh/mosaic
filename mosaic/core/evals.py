from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

def one_cls_f1_score(truth, pred, clss=1):
    findings = truth.columns
    n_findings = len(findings)
    truth = truth.to_numpy()
    pred = pred.to_numpy()
    scores, support = [], []
    for i in range(n_findings):
        t = truth[:,i] == clss
        p = pred[:,i] == clss
        s = f1_score(t, p, average='binary', pos_label=True, zero_division=np.nan)
        st = np.sum(t)
        if st == 0 and s == 0:
            s = np.nan
        scores.append(s), support.append(st)
    return dict(zip(findings, scores)), dict(zip(findings, support))


def weighted_by_support_f1_score(truth, pred):
    findings = truth.columns
    n_findings = len(findings)
    truth = truth.to_numpy()
    pred = pred.to_numpy()
    scores = []
    for i in range(n_findings):
        t = truth[:,i]
        present_classes = np.unique(t) 
        # remove -1 (no mention)
        present_classes = [c for c in present_classes if c != -1] 
        p = pred[:,i]
        s = f1_score(t, p, average='weighted', labels=present_classes, zero_division=np.nan)
        scores.append(s)
    return dict(zip(findings, scores))

def get_F1_scores(truth, pred):
    """"
    Get the F1 scores for each class and the weighted F1 score.
    
    Args:
        truth (pd.DataFrame): truth labels
        pred (pd.DataFrame): predicted labels
        
    Returns:
        pd.DataFrame: F1 scores for each class
        pd.DataFrame: support for each class
        pd.DataFrame: weighted F1 scores
    """
    per_clss_scores = []
    per_clss_support = []

    unique_classes = np.unique(truth)

    for clss in unique_classes:
        findings, support = one_cls_f1_score(truth, pred, clss=clss)
        findings = pd.DataFrame(findings, index=[clss])
        support = pd.DataFrame(support, index=[clss])
        per_clss_scores.append(findings)
        per_clss_support.append(support)
                                
    per_clss_scores = pd.concat(per_clss_scores, axis=0)
    per_clss_support = pd.concat(per_clss_support, axis=0)

    weighted = weighted_by_support_f1_score(truth, pred)
    weighted = pd.DataFrame(weighted, index=[0])

    return per_clss_scores, per_clss_support, weighted
