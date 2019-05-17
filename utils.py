import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
        
def f1_score_bert(preds, true_labels):
    pred_values = np.argmax(preds, axis=1)
    return f1_score(true_labels, pred_values)

def custom_show_top_losses(txt_ci:TextClassificationInterpretation, test_df:pd.DataFrame, text_cols:list, k:int, max_len:int=70)->None:
    """
    Create a tabulation showing the first `k` texts in top_losses along with their prediction, actual,loss, and probability of
    actual class. `max_len` is the maximum number of tokens displayed.
    """
    from IPython.display import display, HTML
    items = []
    tl_val,tl_idx = txt_ci.top_losses()
    for i,idx in enumerate(tl_idx):
        if k <= 0: break
        k -= 1
        tx,cl = txt_ci.data.dl(txt_ci.ds_type).dataset[idx]
        cl = cl.data
        classes = txt_ci.data.classes
        txt = ' '.join(tx.text.split(' ')[:max_len]) if max_len is not None else tx.text
        ori_txt = f"{test_df.loc[idx.item()][text_cols[0]]},{test_df.loc[idx.item()][text_cols[1]]}"
        tmp = [idx.item(), ori_txt, txt, f'{classes[txt_ci.pred_class[idx]]}', f'{classes[cl]}', f'{txt_ci.losses[idx]:.2f}',
               f'{txt_ci.probs[idx][cl]:.2f}']
        items.append(tmp)
    items = np.array(items)
    names = ['Index', 'Original Text', 'Tokenized', 'Prediction', 'Actual', 'Loss', 'Probability']
    df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
    with pd.option_context('display.max_colwidth', -1):
        display(HTML(df.to_html(index=False)))
