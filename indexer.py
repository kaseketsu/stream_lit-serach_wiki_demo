import numpy as np
import faiss
from transformers import BertModel,BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
class Indexer:
    def __init__(self,model_name = 'uer/roberta-base-finetuned-chinanews-chinese',batch_size = 8):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.index = faiss.IndexFlatL2(768)
        self.batch_size = batch_size
        self.url_map = []

    def text_to_vectors(self,text):
        vectors = []
        print(f'text_to_vectors will use {self.device} to launch')
        for i in tqdm(range(0,len(text),self.batch_size)):
            text_list = text[i:i + self.batch_size].tolist()
            try:
                pt_text = self.tokenizer(text_list,return_tensors = 'pt',padding = True,truncation = True,max_length = 512)
            except Exception as e:
                print(f'error processing line{i}:{text_list}')
            pt_text = {k:v.to(self.device) for k,v in pt_text.items()}
            outputs = self.model(**pt_text)
            vec = outputs.pooler_output.detach().cpu().numpy()
            vectors.extend(vec)
        return np.array(vectors)
    def add_to_index(self,text,url):
        vectors = self.text_to_vectors(text)
        self.index.add(vectors)
        self.url_map.extend(url)
    def build_index_from_csv(self,filepath):
        dt = pd.read_csv(filepath)
        self.add_to_index(dt['title'].fillna('404notfound'),dt['url'].fillna('404notfound'))

    def save_index_url(self,index_path,url_path):
        faiss.write_index(self.index,index_path)
        with open(url_path,'w',encoding = 'utf-8')as f:
            for url in self.url_map:
                f.write(url+'\n')


if __name__ == '__main__':
    indexer = Indexer()
    file_path = r'./data/wiki_zh.csv'
    index_path = r'./data/wiki_zh_index.index'
    url_path = r'./data/wiki_zh_url.text'
    indexer.build_index_from_csv(file_path)
    indexer.save_index_url(index_path,url_path)