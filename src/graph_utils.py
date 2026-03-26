from typing import Dict, List
import numpy as np 

def infer_all_relations(predictions:Dict[str, str], tuples:List[List[str]], emotions:List[str]): 
    better_than = set()
    for tup,pred in zip(tuples, predictions): 
        best = pred['most']
        worst = pred['least']
        others = [e for e in tup if e!=best and e!=worst]
        better_than.update({(best, worst)})
        for other in others: 
            better_than.update({(best, other)})
            better_than.update({(other, worst)})
    return better_than


def relations_to_adjacency(relations, e2i):
    arr = np.zeros((len(e2i.keys()), len(e2i.keys())))
    # swap edges
    for r in relations:
        arr[e2i[r[1]], e2i[r[0]]] = 1
    return arr.astype(np.int32)