# Training-Free Text Emotion Tagging via LLM-Based Best-Worst Scaling 

This repository contains code for the paper [Training-Free Text Emotion Tagging via LLM-Based Best-Worst Scaling](https://aclanthology.org/2026.findings-eacl.351/), published in *Findings of the Association for Computational Linguistics: EACL 2026*. 

## Installation 

It is recommended to install the dependencies in a dedicated python venv. 

The required packages can be found in ``requirements.txt``.

We utilize the ``huggingface transformers`` library in combination with ``outlines`` to query LLMs.

## Data Preparation
In order to obtain the data as described in the paper, run ``download_data.py``. 
The dataset is a subset of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions). 

## General Remarks 
The scripts described below all produce output files and place them under specific subdirectories of ``output``. 
Scripts depending on the output of previous scripts typically take references to these outputs (relative to ``output``) as input parameter values. 
All prompts and return format specifications are stored under ``res``.
``eval.py`` can be used to evaluate all kinds of predictions for the dataset. The script ``rescaling.py`` executes the rescaling approach described in the paper and can be applied to all kinds of predictions.
For details on the parameters, see the ``parse_args()`` method at the beginning of each script. 

## Our Method 
The tagging method comprises three steps: 
* query a LLM for Valence/Arousal scores via ``vad_guess.py``. 
* execute LLM-based best-worst scaling via ``bws.py`` 
* based on these decisions, run random walks on the induced graph with ``graphs.py`` 

Example calls: 

``python3 src/vad_guess.py --model allenai/OLMo-2-0325-32B-Instruct --output_file va_predictions_32b``

``python3 src/bws.py --model allenai/OLMo-2-0325-32B-Instruct --dataset go_emotions_sample_test --vad_predictions va_predictions_32b --output_dir 32b_bws`` 

``python3 src/graphs.py --bws_predictions va_predictions_32b/bws_raw.json --dataset go_emotions_sample_test --output_dir rw_results``

## Benchmarks 
Cf. *Experimental Setup* section of the paper. 

### D-A 
(ask for all predictions per text snippet with one prompt)

See ``directly_ask_all.py``. 

Example call: 

``python3 src/directly_ask_all.py --model allenai/OLMo-2-0325-32B-Instruct --dataset go_emotions_sample_test --output_file 32b_all``

### D-I 
(ask for separate predictions per text snippet and emotion, i.e. one emotion and text per prompt)

See ``directly_ask_per_emotion.py``. 

Example call: 

``python3 src/directly_ask_per_emotion.py --model allenai/OLMo-2-0325-32B-Instruct --dataset go_emotions_sample_test --output_file 32b_i``

### Bagdon et al. 2024
The reproduction of [this paper](https://aclanthology.org/2024.naacl-long.439/) comprises two steps: 
* obtain LLM-based best/worst decisions with ``bagdon_et_al.py`` 
* compute the scores on the outputs via ``bagdon_bws_scores.py`` 

Example calls: 

``python3 bagdon_et_al.py --model allenai/OLMo-2-0325-32B-Instruct --dataset go_emotions_sample_test --output_dir 32b_textbws_votes`` 

``python3 bagdon_bws_scores.py --bws_predictions 32b_textbws_votes --output_dir 32b_textbws_scores``

