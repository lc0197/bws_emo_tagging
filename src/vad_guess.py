from argparse import ArgumentParser
from global_vars import *
import pandas as pd
from outlines import generate, Template  
from common import load_text_file, build_sampler, build_hf_model
import json 

def parse_args():
    parser = ArgumentParser() 
    parser.add_argument('--model')
    parser.add_argument('--system_prompt_file', type=str, default='emotion_annotator.txt')
    parser.add_argument('--user_prompt_file', type=str, default='vad')
    parser.add_argument('--dataset', type=str, required=True, help='Used for extracting the emotions to guess VAD values for')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--return_format_file', type=str, required=True)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bit8', action='store_true')
    parser.add_argument('--sampler_type', default=GREEDY, choices=SAMPLERS)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    # parser.add_argument('--save_every', type=int, default=2)
    args = parser.parse_args()
    assert not (args.bit8 and args.fp16), "--bit8 and --fp16 are mutually exclusive."
    return args 
    

if __name__ == '__main__':
    args = parse_args()
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'train.csv'))
    if args.debug_mode: 
        train_df = train_df.iloc[:5, :]
    emotions = list(train_df.columns)[4:]

    output_file = os.path.join(OUT_DIR, 'vad_guess', args.output_file)
    os.makedirs(
        os.path.dirname(output_file), 
        exist_ok=True)

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

    raw_prompt = [{'role': 'system', 'content': system_prompt},
         {'role': 'user', 'content': user_prompt(
             emotions = emotions
         )}]
    formatted_prompt = tokenizer.apply_chat_template(
        raw_prompt, 
        tokenize = False, 
        add_bos = True, 
        add_generation_prompt = True
    )
    
    ret_dict = {'config': vars(args)}
 
    response = generator(formatted_prompt)
    ret_dict['results'] = {'prompt': formatted_prompt, 
                         'response': response} 

    with open(output_file, 'w+') as f:
        json.dump(ret_dict, f)