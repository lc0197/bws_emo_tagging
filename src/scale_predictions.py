import numpy as np 
from argparse import ArgumentParser 
from global_vars import *
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from common import load_predictions
from eval import compute_evaluation

def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--dev_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True) 
    return parser.parse_args()


def find_best_threshold(dev_e_gs, dev_e_preds, start=0., end=1., steps=101): 
    candidates = np.linspace(start=start, stop=end, num=steps)
    best_candidate = None
    best_score = -np.inf
    for candidate in candidates: 
        preds = (dev_e_preds >= candidate).astype(int)
        # avoid degenerate solutions 
        if np.sum(preds) in [0, len(preds)]: 
            continue
        score = f1_score(dev_e_gs, preds)
        if score > best_score: 
            best_score = score
            best_candidate = candidate
    if best_candidate is None: 
        best_candidate = 0.5
    return best_candidate

def scale_predictions(preds, thresh): 
    if thresh < 1: 
        return np.where(preds < thresh, 
                                (preds / thresh) * 0.5, 
                                0.5*(1 + ((preds - thresh) / (1-thresh))))
    else: 
        return np.where(preds == 1, 1., 0.5*preds)

if __name__ == '__main__': 
    args = parse_args()

    out_dir = os.path.join(OUT_DIR, 'rescaled', args.predictions)
    os.makedirs(out_dir, exist_ok=True)

    test_df = pd.read_csv(os.path.join(DATA_DIR, args.test_dataset, 'data.csv'))
    test_df.sort_values(by='ID', inplace=True) 
    test_ids = test_df.ID.values.tolist()
    emotions = list(test_df.columns)[4:]

    dev_df = pd.read_csv(os.path.join(DATA_DIR, args.dev_dataset, 'data.csv'))
    dev_df.sort_values(by='ID', inplace=True) 
    dev_ids = dev_df.ID.values.tolist()

    assert len(set(dev_ids) & set(test_ids)) == 0

    pred_df = load_predictions(args.predictions, emotions=emotions)
    pred_df.sort_values(by='ID', inplace=True)

    dev_pred_df = pred_df[pred_df.ID.isin(dev_ids)]
    assert len(dev_pred_df) == len(dev_df)
    test_pred_df = pred_df[pred_df.ID.isin(test_ids)]
    assert len(test_pred_df) == len(test_df)
    print(f'Found {len(pred_df)} predictions, of which {len(dev_pred_df)} are for the development set and {len(test_pred_df)} are for the test set.')
    
    old_result_df = compute_evaluation(test_df, test_pred_df, emotions)
    print("Before Rescaling: ")
    print(old_result_df)
    print()
    new_test_pred_df = pd.DataFrame(columns=['ID']+emotions)
    new_test_pred_df['ID'] = test_pred_df.ID.values.tolist()

    new_dev_pred_df = pd.DataFrame(columns=['ID']+emotions)
    new_dev_pred_df['ID'] = dev_pred_df.ID.values.tolist()


    for e in emotions: 
        dev_e_preds = dev_pred_df[e].values 
        dev_e_gs = dev_df[e].values
        best_thresh = find_best_threshold(dev_e_gs, dev_e_preds)
        
        scaled_test_preds = scale_predictions(preds=test_pred_df[e].values, thresh=best_thresh)
        scaled_dev_preds = scale_predictions(preds=dev_e_preds, thresh=best_thresh)
        new_test_pred_df[e] = scaled_test_preds
        new_dev_pred_df[e] = scaled_dev_preds

    new_test_pred_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
    new_dev_pred_df.to_csv(os.path.join(out_dir, 'dev_predictions.csv'), index=False)
    
    result_df = compute_evaluation(test_df, new_test_pred_df, emotions)
    print('After rescaling:')
    print(result_df)
    result_df.to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    print() 
    print()