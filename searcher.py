import numpy as np
import faiss
from transformers import BertTokenizer,BertModel
import torch
from tqdm import tqdm
class Searcher:
    def __init__(self,index_path,url_path,model_name = 'uer/roberta-base-finetuned-chinanews-chinese'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index = faiss.read_index(index_path)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.url_map = self.get_url_map(url_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_url_map(self,url_path):
        url_map = []
        with open(url_path,'r',encoding = 'utf-8') as f:
            for line in f:
                url_map.append(line.strip())
        return url_map

    def text_to_vect(self,text):
        input = self.tokenizer(text,return_tensors = 'pt',padding = True,truncation = True,max_length = 512 )
        input = {k:v.to(self.device) for k,v in input.items()}
        output = self.model(**input)
        vec = output.pooler_output.detach().cpu().numpy()
        return vec

    def query(self,text,k = 10):
        vec_text = self.text_to_vect(text)
        dis,index = self.index.search(vec_text,k)
        result = []
        for index in index[0]:
            result.append(self.url_map[index])
        return result

if __name__ == '__main__':
    index_path = r'./data/wiki_zh_index.index'
    url_path = r'./data/wiki_zh_url.text'
    searcher = Searcher(index_path,url_path)
    result = searcher.query('悟空',k = 5)
    for url in tqdm(result):
        print(url)
