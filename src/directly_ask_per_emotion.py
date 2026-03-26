from argparse import ArgumentParser
from global_vars import *
import pandas as pd
from outlines import generate, Template  
from tqdm import tqdm 
from common import load_text_file, build_hf_model, build_sampler
import json 
import itertools 

# D-I 

def parse_args():
    parser = ArgumentParser() 
    parser.add_argument('--model')
    parser.add_argument('--system_prompt_file', type=str, default='emotion_annotator.txt')
    parser.add_argument('--user_prompt_file', type=str, default='ask_directly_per_emotion')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--return_format_file', type=str, default='direct_float_guess_single_emo')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bit8', action='store_true')
    parser.add_argument('--sampler_type', default=GREEDY, choices=SAMPLERS)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--emotion_subset', type=str, nargs='+')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    assert not (args.bit8 and args.fp16), "--bit8 and --fp16 are mutually exclusive."
    return args 
    

if __name__ == '__main__':
    args = parse_args()
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'train.csv'))
    if args.debug_mode: 
        train_df = train_df.iloc[:5, :]
    emotions = list(train_df.columns)[4:]
    if not (args.emotion_subset is None): 
        emotions = [e for e in emotions if e in args.emotion_subset]
    texts = [str(t) for t in train_df[TEXT_COL].values] 
    text_types = train_df[TYPE_COL].values
    ids = train_df[ID_COL].values

    output_file = os.path.join(OUT_DIR, 'directly_ask_per_emo', args.output_file)
    if os.path.exists(output_file) and not args.overwrite: 
        print('Output file exists - not doing this')
        exit(-1)
    os.makedirs(
        os.path.dirname(output_file), 
        exist_ok=True)

    system_prompt = load_text_file(os.path.join(SYSTEM_PROMPT_TEMPLATE_DIR, args.system_prompt_file))
    user_prompt = Template.from_file(os.path.join(USER_PROMPT_TEMPLATE_DIR, args.user_prompt_file))
    return_format = load_text_file(os.path.join(RETURN_TEMPLATE_DIR, args.return_format_file), 
                                   remove_breaks=True)

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

    raw_prompts = list(itertools.chain.from_iterable([
        [[{'role': 'system', 'content': system_prompt},
         {'role': 'user', 'content': user_prompt(
             emotion = emotion, 
             text = t, 
             text_type = ttype
         )}] for emotion in emotions]
        for t,ttype in zip(texts, text_types)
     ]))
    formatted_prompts = [tokenizer.apply_chat_template(
        prompt, 
        tokenize = False, 
        add_bos = True, 
        add_generation_prompt = True
    ) for prompt in raw_prompts]
    
    ret_dict = {'config': vars(args), 'results': []}

    input_emotions = list(itertools.chain.from_iterable([emotions for _ in range(len(ids))]))
    ids = list(itertools.chain.from_iterable([[i for _ in range(len(emotions))] for i in ids]))
    
    for i in tqdm(list(range(0, len(formatted_prompts), args.save_every))): 
        input_prompts = formatted_prompts[i:i+args.save_every]
        responses = generator(input_prompts)
        assert len(responses) == len(input_prompts)
        ret_dict['results'].extend([{'id': ids[i+j], 
                                    'emotion': input_emotions[i+j],
                         'prompt': input_prompts[j], 
                         'response': responses[j]} 
                         for j in range(len(responses))])

        with open(output_file, 'w+') as f:
            json.dump(ret_dict, f)