import numpy as np 
from argparse import ArgumentParser 
from global_vars import *
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from common import load_predictions

def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_csv', required=False)
    return parser.parse_args()


def ece(y_true, preds, num_bins=10): 
    bin_boundaries = np.linspace(0,1,num_bins+1)
    bins = np.digitize(preds, bin_boundaries)
    # if prediction is 1, the bin will be larger than allowed
    bins = np.where(bins < num_bins+1, bins, num_bins)
    N = preds.shape[0]
    ece = 0.0
    # careful, bin indices start at 1
    for i in range(1, num_bins+1):
        mask = (bins == i)
        if np.sum(mask) > 0:
            bin_preds = preds[mask]
            bin_true = y_true[mask]
            avg_pred = np.mean(bin_preds)
            avg_true = np.mean(bin_true)
            ece += (np.sum(mask) / N) * abs(avg_pred - avg_true)
    return ece

def compute_evaluation(train_df, pred_df, emotions): 
    counts = np.sum(train_df.iloc[:, 4:].values, axis=0)
    weights = counts / np.sum(counts)

    eces = {}
    aucs = {}
    f1s = {}
    precisions = {} 
    recalls = {}
    for emotion in emotions: 
        eces[emotion] = ece(y_true = train_df[emotion].values, 
                            preds = pred_df[emotion].values)
        aucs[emotion] = roc_auc_score(y_true=train_df[emotion].values, 
                                    y_score=pred_df[emotion].values)
        f1s[emotion] = f1_score(y_true=train_df[emotion].values, 
                                y_pred=(pred_df[emotion].values >= 0.5).astype(np.int32))
        precisions[emotion] = precision_score(y_true=train_df[emotion].values, 
                                y_pred=(pred_df[emotion].values >= 0.5).astype(np.int32))
        recalls[emotion] = recall_score(y_true=train_df[emotion].values, 
                                y_pred=(pred_df[emotion].values >= 0.5).astype(np.int32))
        
    weighted_ece = np.dot(weights, np.array([eces[e] for e in emotions])) 
    weighted_auc = np.dot(weights, np.array([aucs[e] for e in emotions]))
    weighted_f1 = np.dot(weights,np.array([f1s[e] for e in emotions])) 
    weighted_precision = np.dot(weights,np.array([precisions[e] for e in emotions]))  
    weighted_recall = np.dot(weights,np.array([recalls[e] for e in emotions]))  

    result_df = pd.DataFrame(columns=['emotion', 'ece', 'auc', 'f1', 'precision', 'recall'])
    result_df['emotion'] = emotions + ['mean', 'std', 'weighted_mean']
    result_df['ece'] = [eces[e] for e in emotions] + [np.mean(list(eces.values())), np.std(list(eces.values())), weighted_ece]
    result_df['auc'] = [aucs[e] for e in emotions] + [np.mean(list(aucs.values())), np.std(list(aucs.values())), weighted_auc]
    result_df['f1'] = [f1s[e] for e in emotions] + [np.mean(list(f1s.values())), np.std(list(f1s.values())), weighted_f1]
    result_df['precision'] = [precisions[e] for e in emotions] + [np.mean(list(precisions.values())), np.std(list(precisions.values())), weighted_precision]
    result_df['recall'] = [recalls[e] for e in emotions] + [np.mean(list(recalls.values())), np.std(list(recalls.values())), weighted_recall]

    return result_df 

if __name__ == '__main__': 
    args = parse_args()
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    train_df.sort_values(by='ID', inplace=True)
    print(train_df.ID.values[:5])
    emotions = list(train_df.columns)[4:]

    pred_df = load_predictions(args.predictions, emotions=emotions)
    pred_df.sort_values(by='ID', inplace=True)
    print(pred_df.ID.values[:5])

    overlapping_ids = set(train_df.ID.values.tolist()) & set(pred_df.ID.values.tolist())
    if len(overlapping_ids) != len(train_df) or len(overlapping_ids) != len(pred_df): 
        print(f'Warning: found length mismatch between data and predictions. There are {len(train_df)} IDs in {args.dataset}, {len(pred_df)} IDs in {args.predictions}. Overlap is {len(overlapping_ids)}.')
    
    train_df = train_df[train_df.ID.isin(overlapping_ids)]
    pred_df = pred_df[pred_df.ID.isin(overlapping_ids)] 
    
    result_df = compute_evaluation(train_df, pred_df, emotions)

    if not (args.output_csv is None): 
        out_file = os.path.join(OUT_DIR, 'calibration_eval', args.dataset, args.output_csv)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        result_df.to_csv(out_file, index=False)
    print(result_df)
    
            

        