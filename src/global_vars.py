import os 
from pathlib import Path 

ROOT_DIR = os.path.abspath(Path(__file__).parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_downloaded')
RES_DIR = os.path.join(ROOT_DIR, 'res')
OUT_DIR = os.path.join(ROOT_DIR, 'outputs')
SYSTEM_PROMPT_TEMPLATE_DIR = os.path.join(RES_DIR, 'system_prompts')
USER_PROMPT_TEMPLATE_DIR = os.path.join(RES_DIR, 'user_prompts')
RETURN_TEMPLATE_DIR = os.path.join(RES_DIR, 'return_formats')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

#print('here')
TEXT_COL = 'text'
ID_COL = 'ID'
DS_COL = 'dataset'
TYPE_COL = 'text_type'

GO_EMOTIONS = 'go_emotions'

# sampling methods 
MULTINOMIAL = 'multinomial'
GREEDY = 'greedy'
BEAM = 'beam'
SAMPLERS = [MULTINOMIAL, GREEDY, BEAM]