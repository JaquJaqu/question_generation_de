This repository is a practical implementation of a question generation solution for German texts. 


Requirements & Imports:

T5 Training: 

	pip install ls
	pip install nlp
	pip install --quiet transformers==4.1.1
	pip install  --quiet pytorch-lightning==1.1.3
	pip install --quiet tokenizers==0.9.4 
	pip install --quiet sentencepiece==0.1.94
	pip install --quiet tqdm==4.56.0
	pip install datasets==1.0.2
	pip install tqdm==4.55.1

	from datasets import load_dataset
	from pprint import pprint 
	from tqdm.notebook import tqdm
	from sklearn.utils import shuffle
	from sklearn.model_selection import train_test_split
	from itertools import chain
	from string import punctuation
	from torch.utils.data import Dataset, DataLoader
	from tqdm.auto import tqdm
	from sklearn import metrics
	from pytorch_lightning import Trainer
	from pytorch_lightning.callbacks import EarlyStopping


	from transformers import (
   	 AdamW,
   	 T5ForConditionalGeneration,
    	T5TokenizerFast,
   	 get_linear_schedule_with_warmup
	)
	from pprint import pprint
	from tqdm.notebook import tqdm

	import pandas as pd
	import os
	import sklearn
	import pandas as pd
	import argparse
	import glob
	import os
	import json
	import time
	import logging
	import random
	import re
	import numpy as np
	import torch
	import pytorch_lightning as pl
	import textwrap
	import copy
	import torch
	import textwrap
____________________________________________________________________________________________________________

Question Generation:

	pip install torch torchvision
	pip install spacy==2.1.3 --upgrade --force-reinstall
	pip install -U nltk
	pip install gensim
	pip install git+https://github.com/boudinfl/pke.git
	pip install bert-extractive-summarizer --upgrade --force-reinstall 
	pip install -U pywsd 
	pip install flashtext

	python -m spacy download en/de

	import transformers
	import torch
	import nltk
	import pprint
	import itertools
	import re
	import pke
	import string
	import spacy
	import requests
	import random
	from summarizer import Summarizer
	from transformers import *
	from nltk.corpus import stopwords
	from pywsd.similarity import max_similarity
	from pywsd.lesk import adapted_lesk
	from nltk.corpus import wordnet as wn
	from nltk.tokenize import sent_tokenize
	from flashtext import KeywordProcessor

	nltk.download('stopwords')
	nltk.download('popular')


__________________________________________________________________________________________________________________


# Thanks to 
# [Ramsri Goutham Golla](https://github.com/ramsrigouthamg) 
# [Suraj Patil](https://github.com/patil-suraj) 
# [renatoviolin](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text)
# for their T5 transformer, BERT and QG codes. Their notebooks were instrumental in crafting this. 