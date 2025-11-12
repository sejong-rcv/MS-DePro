import numpy as np

"""
This code is about the strategies for refining proposals. See
Table E. in the supplementary material.
"""

def avg_proposal(proposals):
    avg_bboxs = np.sum(proposals, axis=0) / len(proposals)
    return avg_bboxs

def weighted_proposal(proposals, score):
    weight = score / np.sum(score)        
    out_proposals = np.array(proposals) * weight[:,None]
    out_proposals = np.sum(out_proposal, axis=0)    
    return out_proposals