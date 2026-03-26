from argparse import ArgumentParser
import os 
from global_vars import * 
import json 
import pandas as pd 

# computes scores from raw Bagdon et al. BWS results ((wins-losses)/all)

def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument('--bws_predictions', type=str, required=True, 
                        help='Output directory produced by bagdon_et_al.py script')
    parser.add_argument('--dataset', type=str, default='go_emotions_sample')
    parser.add_argument('--output_dir', default='5_13B', 
                        help="outputs will be stored under outputs/bagdon_scores")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args() 
    return args 

if __name__ == '__main__': 
    args = parse_args() 
    with open(os.path.join(OUT_DIR, 'bagdon_et_al', args.bws_predictions, 'bws_raw.json'), 'r') as f:
        dct = json.load(f)

    output_dir = os.path.join(OUT_DIR, 'bagdon_scores', args.output_dir)
    os.makedirs(output_dir, exist_ok=args.overwrite)

    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    train_df.sort_values(by='ID', inplace=True)
    ids = sorted(list(train_df.ID.values))

    emotions = sorted(list(dct.keys()))
    all_scores = {'ID': ids}

    for e in emotions: 
        bws_records = dct[e]
        wins = {i:0 for i in ids}
        losses = {i:0 for i in ids}
        overall = {i:0 for i in ids}

        for record in bws_records: 
            wins[record['best']] += 1
            losses[record['worst']] += 1
            for i in record['ids']: 
                overall[i] += 1 

        scores = {}
        for i in ids: 
            scores[i] = (wins[i] - losses[i]) / overall[i] 
            scores[i] = (scores[i] +1) / 2 # normalize to [0, 1] 
        
        all_scores[e] = [scores[i] for i in ids]
    
    df = pd.DataFrame(all_scores)
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)