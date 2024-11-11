######################################################
########  Running all of the Performance Metrics    ###
########      TCRBART and TCRT5  Evaluation        ####
#######################################################

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import os

from transformers import T5ForConditionalGeneration
from src.tokenizer import TCRT5Tokenizer
from src.utils import *
from src.dataset import TCRpMHCdataset
from src.constants import *
from src.adapter import *
from src.evaluation import *


### Reproducibility Stuff
# Set a fixed seed for Python's random module
random_seed = 42
random.seed(random_seed)

# Set a fixed seed for NumPy
np.random.seed(random_seed)

# Set a fixed seed for PyTorch
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optionally, set the seed for Hugging Face Transformers as well
from transformers import set_seed as tformer_set_seed
tformer_set_seed(random_seed)

tokenizer = TCRT5Tokenizer().from_pretrained('dkarthikeyan1/tcrt5_ft_tcrdb')
model = T5ForConditionalGeneration.from_pretrained('dkarthikeyan1/tcrt5_ft_tcrdb').to('cuda')

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
        
if __name__ == "__main__":
    mode = 'dataset'
    ### Step 1. Set up the evaluation Dataset using the standard configuration
    topk_dset = 'data/topk_val_df.csv'
    df = pd.read_csv(os.path.join(parent_dir, topk_dset))
    dset = TCRpMHCdataset(source='pmhc', target='tcr', use_pseudo=True, use_cdr3=True, use_mhc=False)
    dset._load_data_from_df(df)
    
    ### Step 2. Identify the model type and crank out model's evaluation
    all_metrics = pd.DataFrame()
    evaluator = ModelEvaluator(tokenizer, model, device='cuda')
    # Get the dataset level metrics
    dataset_metrics = evaluator.dataset_metrics_at_k(dset, k=100, max_len=25, return_translations=False, mode='beam', num_beams=300)
    dataset_metrics_df = pd.DataFrame(dataset_metrics, index=["TCRT5"])
    # Get the per pMHC level metrics
    atomic_metrics = evaluator.atomic_metrics_at_k(dset, k=100, max_len=25, return_translations=True, mode='beam', num_beams=300)
    atomic_metrics_simplified = {k.peptide+'_'+k.allele: v for k,v in atomic_metrics.items()}
    atomic_metrics_df = pd.DataFrame(atomic_metrics_simplified).T.sort_index()
    atomic_metrics_df['model'] = "TCRT5"
    
    ### Step 3. Clean up the memory
    del model # Free up GPU
    del evaluator # Reset the model
    del tokenizer # Reset Tokenizer
    torch.cuda.empty_cache()
    
    ### Step 4. Save the metrics to a CSV file
    dataset_metrics_df.to_csv(f'tcrt5_dataset_eval.csv')
    atomic_metrics_df.to_csv(f'tcrt5_atomic_eval.csv')
    print("Reached the end.")
