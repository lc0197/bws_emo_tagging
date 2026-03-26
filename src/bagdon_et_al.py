from argparse import ArgumentParser
from global_vars import *
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from common import load_text_file, build_hf_model, build_sampler
from outlines import generate, Template  
import itertools 

# performs LLM-based BWS as proposed by Bagdon et al. 2024\

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    # prompting parameters
    parser.add_argument('--system_prompt_file', type=str, default='emotion_annotator.txt')
    parser.add_argument('--user_prompt_file', type=str, default='bws_texts')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--return_format_file', type=str, default='bws_guess_texts.json')
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
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_factor', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    # bws parameters 
    parser.add_argument('--bws_sample_size', type=int, default=6, help='parameter M in the paper')
   # output 
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    return args 


def do_raw_bws_prompts(args, bws_blocks, raw_save_file, emotions, batch_size=4):

    # initialize prompts, model, generators
    system_prompt = load_text_file(os.path.join(SYSTEM_PROMPT_TEMPLATE_DIR, args.system_prompt_file))
    user_prompt = Template.from_file(os.path.join(USER_PROMPT_TEMPLATE_DIR, args.user_prompt_file))

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

    results = {e:[] for e in emotions}

    for e in tqdm(emotions): 
        print(f'Prompting for {e}')
        # generate BWS prompts
        raw_prompts = [
                [{'role': 'system', 'content': system_prompt},
                 {'role': 'user', 'content': user_prompt(
                     emotion = e, 
                     texts = block[1], 
                     text_type = block[2][0] # assuming all have the same text type
                     )}] 
                     for block in bws_blocks
                     ]
        
        formatted_prompts = [tokenizer.apply_chat_template(
            prompt, 
            tokenize = False, 
            add_bos = True, 
            add_generation_prompt = True) 
            for prompt in raw_prompts]


        for i in tqdm(list(range(0, len(bws_blocks), batch_size))):
            batch_prompts = formatted_prompts[i:i+batch_size]
            batch_blocks = bws_blocks[i:i+batch_size]
            responses = generator(batch_prompts)
            
            for k in range(len(batch_blocks)):
                ids = batch_blocks[k][0]
                texts = batch_blocks[k][1]
                best = responses[k]['most']
                best_id = ids["ABCDEFGHIJ".index(best)]
                worst = responses[k]['least']
                worst_id = ids["ABCDEFGHIJ".index(worst)]
                results[e].append({
                    'ids': ids, 
                    'best': best_id,
                    'worst': worst_id 
                    #'prompt': batch_prompts[k]
                })
                

        with open(raw_save_file, 'w+') as f: 
            json.dump(results, f)

    return results


if __name__ == '__main__': 
    args = parse_args()
    # load data 
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'data.csv'))
    emotions = list(train_df.columns)[4:]
    texts = [str(x) for x in train_df.text.values]
    text_types = train_df['text_type'].values.tolist() 
    ids = train_df['ID'].values


    # build text blocks randomly
    generator = np.random.default_rng(args.random_seed)
    idxs = list(range(len(train_df)))
    num_repeats = args.sample_factor * args.bws_sample_size
    idxs = list(itertools.chain.from_iterable(
        [generator.permutation(idxs).tolist() 
         for _ in range(num_repeats)]))
    
    idx_blocks = [idxs[i:i+args.bws_sample_size] for i in range(0, len(idxs), args.bws_sample_size)]
    id_blocks = [[ids[i] for i in idx_block] for idx_block in idx_blocks]
    text_blocks = [[texts[i] for i in idx_block] for idx_block in idx_blocks]
    text_type_blocks = [[text_types[i] for i in idx_block] for idx_block in idx_blocks]
    bws_blocks = [(id_blocks[i], text_blocks[i], text_type_blocks[i]) for i in range(len(id_blocks))]
    print('BWS blocks', len(bws_blocks))
    if args.debug_mode: 
        bws_blocks = bws_blocks[:5]
    

    output_dir = os.path.join(OUT_DIR, 'bagdon_et_al', args.output_dir)
    os.makedirs(output_dir, exist_ok=args.overwrite)
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    
    raw_out = os.path.join(output_dir, 'bws_raw.json')
    bws_results_raw = do_raw_bws_prompts(args, 
                             bws_blocks=bws_blocks,
                             emotions=emotions, 
                             raw_save_file=raw_out, 
                             batch_size= args.batch_size)
