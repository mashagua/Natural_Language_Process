import pandas as pd
import nltk
# 读取原始的训练数据及测试数据
train_data_ori = pd.read_csv("../Data/train (1).csv",header=None)
test_data_ori = pd.read_csv("../Data/test.csv",header=None)
with open("../Data/eng-stopwords.txt","r") as f:
    STOPWORDS = set([line.strip() for line in f.readlines()]) # 英语停用词


nltk.download('punkt')  # 下载英文分词所需拓展包
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer  # 导入 nltk.stem 用来词形还原

WML = WordNetLemmatizer()  # 词形还原器

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith(''):
        return  wordnet.ADV
    else:
        return wordnet.NOUN
