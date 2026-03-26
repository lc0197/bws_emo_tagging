import pandas as pd
import json 
import numpy as np
from typing import Dict, List, Set, Tuple
from argparse import ArgumentParser
from global_vars import *
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from collections import Counter 
from eval import ece, compute_evaluation
from graph_utils import infer_all_relations, relations_to_adjacency

def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument('--bws_predictions', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='go_emotions_sample_validation')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--only_one_walk', action='store_true', help='only for ablation: stop after one walk.')
    args = parser.parse_args() 
    return args 


# random walk that only selects one node and then stops
def single_random_walk(graph, generator, importance): 
    num_nodes = graph.shape[0]
    encountered_nodes = []
    node = generator.choice(num_nodes)  
    while True:
        # normalize pstop 
        encountered_nodes.append(node)
        if np.sum(graph[node,:]) == 0: # no successors 
            return node, encountered_nodes
         
        successors_importance = np.dot(graph[node,:], importance)
        
        pstop = importance[node] / (importance[node] + successors_importance)

        if generator.random() < pstop:
            return node, encountered_nodes
        else:
            # sample next 
            neighbors = np.argwhere(graph[node] > 0).flatten()
            neighbor_weights = importance[neighbors]
            node = generator.choice(neighbors, p=neighbor_weights/np.sum(neighbor_weights))


def do_single_walks(graph, generator, importance, num_walks=100):
    nodes = []
    for _ in range(num_walks):
        node,_ = single_random_walk(graph, generator, importance)
        nodes.append(node)
    counts = Counter(nodes)
    values = np.array([counts[i]/num_walks if i in counts.keys() else 0 for i in range(graph.shape[0])])
    return values

# random walk according to the modified PageRank model
def random_walk2(graph, generator, only_one_walk=False):
    sampled_nodes = []
    encountered_nodes = []
    importance = np.sum(graph, axis=0) / (np.sum(graph, axis=0) + np.sum(graph, axis=1))
    initial_relative_importance = do_single_walks(graph, generator, importance)

    while len(sampled_nodes) < graph.shape[0]:
        node, enc = single_random_walk(graph, generator, importance)
        encountered_nodes.extend(enc)
        sampled_nodes.append(node)

        in_edges = np.argwhere(graph[:,node] > 0).flatten()
        out_edges = np.argwhere(graph[node,:] > 0).flatten()
        for inp in in_edges:
            for outp in out_edges:
                graph[inp, outp] = 1
        graph = graph.clip(0, 1)
        graph[node, :] = 0
        graph[:, node] = 0
        # stop labelling process here? 
        global_stop_prob = np.sum(initial_relative_importance[sampled_nodes])
        if generator.random() < global_stop_prob or only_one_walk:
            break
            # if not, restart
        importance = np.nan_to_num(np.sum(graph, axis=0) / (np.sum(graph, axis=0) + np.sum(graph, axis=1)), 0)
    return sampled_nodes, encountered_nodes


def execute_random_walks(graph, generator, num_walks=1000, only_one_walk=False):
    all_samples = []
    all_encountered = []
    for _ in range(num_walks):
        samples, encountered = random_walk2(graph.copy(), generator, only_one_walk=only_one_walk)
        all_samples.extend(samples)
        all_encountered.extend(encountered)
    return all_samples, all_encountered


def get_rw_predictions(relation, emotions, generator, num_walks=1000, only_one_walk=False):
    e2i = {e:i for i,e in enumerate(emotions)}
    graph = relations_to_adjacency(relation, e2i=e2i)
 
    sampled_nodes, encountered_nodes = execute_random_walks(generator=generator, 
                                         graph=graph, 
                                         num_walks=num_walks, 
                                         only_one_walk=only_one_walk)
    counts = Counter(sampled_nodes)
    for i in range(len(emotions)):
        if not (i in counts.keys()): 
            counts[i] = 0
    values = [counts[i] for i in range(len(emotions))]
    values = np.array(values) / num_walks 
    encountered = Counter(encountered_nodes)
    for i in range(len(emotions)):
        if not (i in counts.keys()): 
            encountered[i] = 0
    return values, encountered


if __name__ == '__main__': 
    args = parse_args()
    with open(os.path.join(OUT_DIR, args.bws_predictions), 'r') as f:
        dct = json.load(f)

    output_dir = os.path.join(OUT_DIR, 'graphs', args.dataset, args.output_dir)
    os.makedirs(output_dir, exist_ok=args.overwrite)

    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    train_df.sort_values(by='ID', inplace=True)
    emotions = list(train_df.columns)[4:]
    e2i = {e:i for i,e in enumerate(emotions)}
    i2e = {i:e for i,e in enumerate(emotions)}

    ids = sorted(list(
        set(dct.keys()) & set(train_df['ID'].values) 
        ))
    if len(ids) != len(dct.keys()) | len(ids) != len(train_df):
        print(f'Warning: {len(train_df)} IDs in the given dataset. {len(dct.keys())} IDs in the given predictions. Overlap is {len(ids)}.') 
    predictions = [dct[i]['responses'] for i in ids]
    tuples = [dct[i]['bws_samples'] for i in ids]

    relations = [infer_all_relations(predictions=predictions[i], emotions=emotions, tuples=tuples[i]) for i in range(len(ids))]

    generator = np.random.default_rng(42)

    # for test purposes
    id_df = pd.DataFrame({'ID': ids})
    predictions = []
    encountered = []
    for r in tqdm(relations): 
        preds, enc = get_rw_predictions(r, 
                               emotions, 
                               generator, 
                               num_walks=args.num_walks, 
                               only_one_walk = args.only_one_walk)
        
        predictions.append(preds)
        encountered.append(enc)
    pred_arr = np.vstack(predictions)
    pred_df = pd.DataFrame(columns=emotions, data = pred_arr)
    pred_df = pd.concat([id_df, pred_df], axis='columns')
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    enc_dict = {} 
    for i in range(len(ids)): 
        enc_dict[ids[i]] = {i2e[i]: count for i,count in encountered[i].items()} 

    with open(os.path.join(output_dir, 'encountered.json'), 'w+') as f: 
        json.dump(enc_dict, f)

    result_df = compute_evaluation(train_df, pred_df, emotions)
    result_df.to_csv(os.path.join(output_dir, 'eval_results.csv'), index=False)

    