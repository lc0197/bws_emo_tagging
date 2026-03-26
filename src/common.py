from global_vars import *
import torch 
from transformers import AutoTokenizer, BitsAndBytesConfig
from outlines import models, samplers 
import pandas as pd 
import json
import numpy as np


def load_text_file(file, remove_breaks=False):
    with open(file, 'r') as f: 
        lines = f.readlines()
    if remove_breaks:
        lines = [l.replace('\n', '') for l in lines]
    return ''.join(lines)


# auxiliary method for outlines
def build_hf_model(model, fp16, bit8, gpu_id): 
    assert not (fp16 and bit8)
    tokenizer = AutoTokenizer.from_pretrained(model)
    kwargs = {}
    if fp16: 
        kwargs['torch_dtype'] = torch.float16 
    elif bit8: 
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    if torch.cuda.is_available():
        if gpu_id != 'auto':
            device = f'cuda:{gpu_id}'
            model = models.transformers(model, device=device, model_kwargs=kwargs)
        else: 
            kwargs['device_map'] = 'auto'
            model = models.transformers(model, model_kwargs=kwargs)
    else:
        device = 'cpu'
        model = models.transformers(model, device=device, model_kwargs=kwargs)
    # llm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    # llm.to(device)
    
    return model, tokenizer


# auxiliary method for outlines
def build_sampler(sampler_type, temperature, top_k, top_p, num_beams):
    if sampler_type == GREEDY: 
        return samplers.greedy() 
    elif sampler_type == MULTINOMIAL: 
        return samplers.multinomial(top_k=top_k, top_p=top_p, temperature=temperature)
    elif sampler_type == BEAM: 
        return samplers.beam_search(num_beams=num_beams)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
    
# different scripts create different output (formats). This method loads predictions uniformly
def load_predictions(predictions, emotions): 
    # check which type of file 
    # output of ask_directly
    pred_type = os.path.normpath(predictions).split(os.path.sep)[0]
    print(f'Loading predictions of type {pred_type}')
    if pred_type == 'directly_ask': 
        with open(os.path.join(OUT_DIR, predictions), 'r') as f: 
            dct = json.load(f)
        pred_dict = {"ID": []}
        pred_dict.update({e:[] for e in emotions})
        for record in dct['results']: 
            pred_dict['ID'].append(record['id'])
            # there may be errors in following the instruction, e.g. naming emotions twice or not all  
            individual_pred_dict = {e:0 for e in emotions}
            for x in record['response']: 
                individual_pred_dict[x['emotion']] = max(0, min(x['score'], 1))
            for e in emotions: 
                pred_dict[e].append(np.round(individual_pred_dict[e], 4))
        pred_df = pd.DataFrame(pred_dict)
        return pred_df
    # output of ask_directly_per_emotion
    elif pred_type == 'directly_ask_per_emo':
        with open(os.path.join(OUT_DIR, predictions), 'r') as f: 
            dct = json.load(f)
        pred_dict = {"ID": []}
        pred_dict.update({e:[] for e in emotions})
        curr_id = None
        for record in dct['results']:
            if record['id'] != curr_id: 
                curr_id = record['id'] 
                pred_dict['ID'].append(curr_id)
            emotion = record['emotion']
            pred_dict[emotion].append(max(0, min(record['response']['score'], 1)))
        pred_df = pd.DataFrame(pred_dict)
        return pred_df
    elif pred_type == 'graphs' or pred_type == 'bagdon_scores': 
        pred_df = pd.read_csv(os.path.join(OUT_DIR, predictions, 'predictions.csv'))
        return pred_df
    elif pred_type == 'rescaled': 
        dev_df = pd.read_csv(os.path.join(OUT_DIR, predictions, 'dev_predictions.csv'))
        test_df = pd.read_csv(os.path.join(OUT_DIR, predictions, 'test_predictions.csv')) 
        df = pd.concat([dev_df, test_df], ignore_index=True, axis='rows') 
        df.sort_values(by='ID', inplace=True)
        return df
    else: 
        raise NotImplementedError()