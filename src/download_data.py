import os 
import subprocess
from global_vars import GO_EMOTIONS, RAW_DATA_DIR, DATA_DIR, TEXT_COL, ID_COL, DS_COL, TYPE_COL, RES_DIR
import pandas as pd
from tqdm import tqdm
import numpy as np 

print(GO_EMOTIONS)

# Download the raw datasets 

 # for datasets without predefined train-dev-test split
DEFAULT_TRAIN_SIZE = 0.7
SPLIT_SEED = 1234

data_download_dirs = {
    GO_EMOTIONS: 'https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data',
}

dataset_links = {
    GO_EMOTIONS: [data_download_dirs[GO_EMOTIONS]+'/'+f for f in [
        'ekman_mapping.json', 'emotions.txt', 'dev.tsv', 'train.tsv', 'test.tsv'
    ]]
}

def split_df(df, ratio=DEFAULT_TRAIN_SIZE, seed=SPLIT_SEED):
    train = df.sample(frac=ratio, replace=False, random_state=seed)
    rest = df.drop(train.index)
    dev = rest.sample(frac=0.5, replace=False, random_state=seed)
    test = df.drop(dev.index).drop(train.index)
    return train, dev, test


for dataset, links in dataset_links.items():
    tgt_dir = os.path.abspath(os.path.join(RAW_DATA_DIR, dataset)) 
    os.makedirs(tgt_dir, exist_ok=True)
    for link in links:
        tgt_file = os.path.join(tgt_dir, link[link.rfind('/')+1:])
        if not os.path.exists(tgt_file): 
            print('Downloading', link)
            ret = subprocess.run(f'wget {link} -P {tgt_dir}', shell=True)
            if ret.returncode != 0: 
                print('Download failed:', link)
                print(ret)
        else:
            print('File already exists:', tgt_file)
       

def handle_go_emotions():
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, GO_EMOTIONS, 'train.tsv'), sep='\t', header=None)
    dev_df = pd.read_csv(os.path.join(RAW_DATA_DIR, GO_EMOTIONS, 'dev.tsv'), sep='\t', header=None)
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, GO_EMOTIONS, 'test.tsv'), sep='\t', header=None)
    train_df['partition'] = 'train'
    dev_df['partition'] = 'dev'
    test_df['partition'] = 'test'
    #print(train_df.head())
    full_df = pd.concat([train_df, dev_df, test_df], axis='rows')
    full_df.columns = ['text', 'emotion_ids', 'post_id', 'partition']
    with open(os.path.join(RAW_DATA_DIR, GO_EMOTIONS, 'emotions.txt'), 'r') as f:
        emotions = [l.replace('\n', '') for l in f.readlines()]
        i2e = {i:e for i,e in enumerate(emotions)}
        e2i = {e:i for i,e in enumerate(emotions)}
    emotions = sorted(emotions)
    for e in emotions:
        full_df[e] = [int(e2i[e] in [int(i) for i in row['emotion_ids'].split(",")]) for _,row in full_df.iterrows()]
    assert len(set(full_df.post_id.values)) == len(full_df)
    full_df[DS_COL] = GO_EMOTIONS
    full_df[ID_COL] = [f'{GO_EMOTIONS}_{row.post_id}' for _,row in full_df.iterrows()] 
    full_df[TYPE_COL] = 'Reddit post'
    full_df.rename(columns={'text': TEXT_COL}, inplace=True)
    full_df.drop(columns=['emotion_ids', 'post_id'], inplace=True)
    full_df = full_df[[DS_COL, ID_COL, TYPE_COL, TEXT_COL, 'partition'] + emotions]
    # restore train, dev, test
    train_df = full_df[full_df.partition=='train'].copy()
    dev_df = full_df[full_df.partition=='dev'].copy()
    test_df = full_df[full_df.partition=='test'].copy()
    for df in [train_df, dev_df, test_df]:
        df.drop(columns=['partition'], inplace=True)
    return train_df, dev_df, test_df

prep_map = {
    GO_EMOTIONS: handle_go_emotions
}


for ds, func in tqdm(prep_map.items()):
    print('Preparing', ds)
    tgt_dir = os.path.join(DATA_DIR, ds)
    os.makedirs(tgt_dir, exist_ok=True)
    train_df, dev_df, test_df = func()
    assert list(train_df.columns) == list(dev_df.columns)
    assert list(dev_df.columns) == list(test_df.columns)
    assert len( set(train_df[ID_COL].values) & set(dev_df[ID_COL].values) ) == 0
    assert len( set(dev_df[ID_COL].values) & set(test_df[ID_COL].values) ) == 0
    assert len( set(train_df[ID_COL].values) & set(test_df[ID_COL].values) ) == 0
    #print(f'Found {len(train_df)} training samples')
    #print(f'Found {len(dev_df)} development samples')
    #print(f'Found {len(test_df)} test samples')
    for part, df in zip(['train', 'dev', 'test'], [train_df, dev_df, test_df]):
        df.to_csv(os.path.join(tgt_dir, f'{part}.csv'), index=False)
    

# generate validation and test set as used in the paper
go_emotions_train_df = pd.read_csv(os.path.join(DATA_DIR, "go_emotions", "train.csv"))
for partition in ['validation', 'test']:
    with open(os.path.join(RES_DIR, 'data_ids', f'{partition}.txt'), 'r') as f: 
        ids = f.read().splitlines()
        part_df = go_emotions_train_df[go_emotions_train_df.ID.isin(ids)]
        tgt_dir = os.path.join(DATA_DIR, f'go_emotions_sample_{partition}')
        os.makedirs(tgt_dir, exist_ok=True)
        part_df.to_csv(os.path.join(tgt_dir, 'data.csv'), index=False)