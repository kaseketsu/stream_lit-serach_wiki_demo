import json
import csv
import os
def process_line(line,writer):
    try:
        line = json.loads(line)
        title = line.get('title','')
        url = line.get('url','')
        writer.writerow([title,url])
    except json.JSONDecodeError:
        print('该行无法解析')

def process_file(json_file_path,csv_writer):
    with open(json_file_path,'r',encoding = 'utf-8') as file:
        for line in file:
            process_line(line,csv_writer)
def process_directory(file_directory,out_put_path,num_file = 10000):
    with open(out_put_path,'w',encoding = 'utf-8',newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['title','url'])
        file_count = 0
        for root,dir,files in os.walk(file_directory):
            for file in files:
                file_count += 1
                json_file_path = os.path.join(root,file)
                process_file(json_file_path,writer)
                print(f'处理进程:{file_count}/{num_file},正在处理:{json_file_path}')
                if file_count > num_file:
                    return

if __name__ == '__main__':
    wiki_directory = './data/wiki_zh' #wiki数据的路径
    out_put_csv = './data/wiki_zh.csv' #输出文件路径
    num_file = 10000 #处理的数据数量
    process_directory(wiki_directory,out_put_csv,num_file)