from argparse import ArgumentParser
from global_vars import *
import pandas as pd
from tqdm import tqdm
import json
from typing import Tuple, Dict
import numpy as np
from common import load_text_file, build_hf_model, build_sampler
from outlines import generate, Template  
from sklearn.preprocessing import MinMaxScaler
from k_means_constrained import KMeansConstrained
import itertools 

# computes raw scores based on BWS tuples consisting of emotions
# Prerequisite: V/A predictions as obtained by vad_guess.py

# for ablations: optionally, use only V or A clusters. Alternativelz, cluster based on V and A
V = 'v'
VA = 'va'
V_A = 'v_a'
A = 'a'
CLUSTER_MODES = [V, VA, V_A, A]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    # prompting parameters
    parser.add_argument('--system_prompt_file', type=str, default='emotion_annotator.txt')
    parser.add_argument('--user_prompt_file', type=str, default='bws')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--return_format_file', type=str, default='bws_guess')
    parser.add_argument('--gpu_id', type=str, default='0')
    # sampling parameters
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bit8', action='store_true')
    parser.add_argument('--sampler_type', default=GREEDY, choices=SAMPLERS)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)
    #parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--cluster_mode', type=str, choices=CLUSTER_MODES, default=V_A)
    # bws parameters 
    parser.add_argument('--bws_sample_size', type=int, default=4, help='Roughly desired tuple size (M in the paper)')
    parser.add_argument('--random_clustering', action='store_true', help='mutually exclusive with no clustering')
    parser.add_argument('--random_seed', type=int, default=42, help='only used for --random_clustering')
    parser.add_argument('--vad_predictions', type=str, required=False, help='If clustering, predictions for valence/arousal/dominance are needed.')
    
    # output 
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    
    assert (not (args.vad_predictions is None)) or (args.random_clustering)
    if args.random_clustering and not (args.vad_predictions is None):
        print("Warning: --vad_predictions will be ignored because --random_clustering was specified.")
    return args 


def load_vad(vad_predictions_file:str): 
    with open(os.path.join(OUT_DIR, 'vad_guess', vad_predictions_file), 'r') as f:
        vad_data = json.load(f)
    return {response['emotion']:
            tuple([response[dim] for 
                   dim in ['valence', 'arousal', 'dominance']]) for 
                   response in vad_data['results']['response']}


# takes clusters and generates new tuples by combining elements from different clusters
def distribute_to_dissimilar(clusters): 
    num_clusters = len(clusters) 
    dissim_clusters = [[] for _ in range(num_clusters)]
    flattened_clusters = list(itertools.chain.from_iterable(clusters)) 
    for i,e in enumerate(flattened_clusters): 
        cluster_idx = i % num_clusters 
        dissim_clusters[cluster_idx].append(e)
    return dissim_clusters


def simple_clustering(vad_predictions:Dict[str, Tuple[float, float, float]], num_clusters:int, cluster_mode:str): 
    emotions = sorted(list(vad_predictions.keys()))
    arr = np.vstack([np.array(vad_predictions[e]) for e in emotions])
    arr = MinMaxScaler().fit_transform(arr)
    expected_cluster_size = arr.shape[0] // num_clusters
    v_arr = arr[:, 0]
    a_arr = arr[:, 1]
    va_arr = arr[:, :2]
    
    v_close = KMeansConstrained(n_clusters=num_clusters, 
                                size_max= expected_cluster_size+2, 
                                random_state=42).fit_predict(v_arr.reshape(-1, 1))
    a_close = KMeansConstrained(n_clusters=num_clusters, 
                                size_max= expected_cluster_size+2, 
                                random_state=42).fit_predict(a_arr.reshape(-1, 1))
    va_close = KMeansConstrained(n_clusters=num_clusters, 
                                size_max= expected_cluster_size+2, 
                                random_state=42).fit_predict(va_arr)
    
    v_close_clusters = [[] for _ in range(num_clusters)] 
    a_close_clusters = [[] for _ in range(num_clusters)]
    va_close_clusters = [[] for _ in range(num_clusters)]
    for i,e in enumerate(emotions): 
            v_close_clusters[v_close[i]].append(e)
            a_close_clusters[a_close[i]].append(e)
            va_close_clusters[va_close[i]].append(e) 
    v_dist_clusters = distribute_to_dissimilar(v_close_clusters) 
    a_dist_clusters = distribute_to_dissimilar(a_close_clusters)
    va_dist_clusters = distribute_to_dissimilar(va_close_clusters)
    
    if cluster_mode == V_A:
        return v_close_clusters + a_close_clusters + v_dist_clusters + a_dist_clusters  
    elif cluster_mode == V:
        return v_close_clusters + v_dist_clusters
    elif cluster_mode == VA:
        return va_close_clusters + va_dist_clusters
    elif cluster_mode == A: 
        return a_close_clusters + a_dist_clusters
    else:
        raise ValueError("Invalid cluster mode")


def do_raw_bws_prompts(args, vad_predictions=None, clusters=None, raw_save_file=None, random_generator = None, cluster_mode=VA):
    assert not (args.random_clustering and not (clusters is None))
    # load data 
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    if args.debug_mode: 
        train_df = train_df.iloc[:1, :]
    emotions = list(train_df.columns)[4:]
    texts = [str(t) for t in train_df[TEXT_COL].values] 
    text_types = train_df[TYPE_COL].values
    ids = train_df[ID_COL].values

    # initialize prompts, model, generators
    system_prompt = load_text_file(os.path.join(SYSTEM_PROMPT_TEMPLATE_DIR, args.system_prompt_file))
    user_prompt = Template.from_file(os.path.join(USER_PROMPT_TEMPLATE_DIR, args.user_prompt_file))
    #return_format = load_text_file(os.path.join(RETURN_TEMPLATE_DIR, args.return_format_file), 
    #                               remove_breaks=True)
    return_format = Template.from_file(os.path.join(RETURN_TEMPLATE_DIR, args.return_format_file))
    return_format = return_format(emotions=emotions).replace('\n', "")

    model, tokenizer = build_hf_model(
        model= args.model, 
        fp16=args.fp16,
        bit8=args.bit8,
        gpu_id=args.gpu_id
    )

    sampler = build_sampler(
        sampler_type=args.sampler_type,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams
    )
    generator = generate.json(model, return_format, sampler=sampler)

    results = {}

    for i in tqdm(list(range(len(texts)))):
        text = texts[i] 
        id = ids[i]
        text_type = text_types[i]

        if args.random_clustering: 
            random_vad_array = random_generator.random(size=(len(emotions), 3))
            vad_predictions = {e:random_vad_array[i] for i,e in enumerate(emotions)}
            vad_predictions = {k:v for k,v in vad_predictions.items() if k!= 'neutral'}
            num_clusters = (len(emotions)-1) // (args.bws_sample_size)
            clusters = simple_clustering(vad_predictions, num_clusters=num_clusters, cluster_mode=cluster_mode)
        # otherwise, just use the given clusters as BWS samples
        bws_samples = [c+['neutral'] for c in clusters] 

        # generate BWS prompts
        raw_prompts = [
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt(
                 emotions = bws_sample, 
                 text = text, 
                 text_type = text_type
                 )}] 
                 for bws_sample in bws_samples
                 ]
        formatted_prompts = [tokenizer.apply_chat_template(
            prompt, 
            tokenize = False, 
            add_bos = True, 
            add_generation_prompt = True) 
            for prompt in raw_prompts]
        
        # pass them through the model and store
        responses = generator(formatted_prompts)
        results[id] = {'bws_samples': bws_samples, 
                       'prompts': formatted_prompts, 
                       'responses': responses}
        if raw_save_file: 
            with open(raw_save_file, 'w+') as f: 
                json.dump(results, f)

    return results


        
if __name__ == '__main__': 
    args = parse_args()
    # load data 
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    emotions = list(train_df.columns)[4:]

    output_dir = os.path.join(OUT_DIR, 'bws', args.dataset, args.output_dir)
    os.makedirs(output_dir, exist_ok=args.overwrite)
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # load VAD and clustering, if necessary 

    if not args.random_clustering:
        vad_predictions = load_vad(args.vad_predictions)
        vad_predictions = {k:v for k,v in vad_predictions.items() if k != 'neutral'}

        num_clusters = (len(emotions)-1) // (args.bws_sample_size)  # -1 because of neutral
        
        clusters = simple_clustering(vad_predictions, num_clusters=num_clusters, cluster_mode=args.cluster_mode)
    else: 
        vad_predictions = None 
        clusters = None 

    raw_out = os.path.join(output_dir, 'bws_raw.json')
    bws_results_raw = do_raw_bws_prompts(args, 
                         vad_predictions=vad_predictions, 
                         clusters=clusters, 
                         raw_save_file=raw_out, 
                         random_generator= np.random.default_rng(args.random_seed), 
                         cluster_mode=args.cluster_mode)
