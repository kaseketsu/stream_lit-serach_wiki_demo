import streamlit as st
from searcher import Searcher
st.title('花少的搜索引擎')
input = st.text_input('请输入搜索内容')


index_path = r'./data/wiki_zh_index.index'
url_path = r'./data/wiki_zh_url.text'
searcher = Searcher(index_path,url_path)
if input:
    result = searcher.query(input)
    for url in result:
        st.write(url)
else:
    st.info('内容不能为空')

