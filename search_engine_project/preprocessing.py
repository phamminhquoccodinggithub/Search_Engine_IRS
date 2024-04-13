# Import
import os
import re
import numpy as np
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

DIR_PATH = 'dataset/'

def get_topics(dir_path=DIR_PATH):
    return sorted(os.listdir(dir_path))

def preprocessing(doc):
    # Tiến hành xử lý các lỗi từ/câu, dấu câu, v.v. trong tiếng Việt với hàm text_normalize
    normalized_doc = text_normalize(doc)
    # Tiến hành tách từ
    tokens = word_tokenize(normalized_doc)
    # Tiến hành kết hợp các từ ghép trong tiếng Việt bằng '_'
    combined_tokens = [token.replace(' ', '_') for token in tokens]
    return (normalized_doc, combined_tokens)


def fetch_doc_by_topic(topic):
    data_root_dir_path = DIR_PATH + topic
    docs = []
    files = os.listdir(data_root_dir_path)
    for file_name in files:
        file_path = os.path.join(data_root_dir_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.lower().strip()
                lines.append(line)
        doc = " ".join(lines)
        clean_doc = re.sub(r'\W+',' ', doc)
        (normalized_doc, tokens) = preprocessing(clean_doc)
        docs.append((topic, normalized_doc, tokens))
    return docs


# Cấu trúc dữ liệu dictionary để lưu thông tin chủ đề-tài liệu, nhằm hỗ trợ cho việc tìm kiếm nhanh
topic_doc_idxes_dict = {}
doc_idx_topic_dict = {}
D = []

# Duyệt qua từng chủ đề
doc_idx = 0
topics = get_topics()
for topic in topics:
    current_topic_docs = fetch_doc_by_topic(topic)
    topic_doc_idxes_dict[topic] = []
    for (topic, normalized_doc, tokens) in current_topic_docs:
        topic_doc_idxes_dict[topic].append(doc_idx)
        doc_idx_topic_dict[doc_idx] = topic
        doc_idx+=1
    D += current_topic_docs

doc_size = len(D)
print('Hoàn tất, tổng số lượng tài liệu/văn bản đã lấy: [{}]'.format(doc_size))
for topic in topic_doc_idxes_dict.keys():
    print(' - Chủ đề [{}] có [{}] tài liệu/văn bản.'.format(topic, len(topic_doc_idxes_dict[topic])))
    
import pickle

# save the iris classification model as a pickle file
index_file = "index.pkl"  

with open(index_file, 'wb') as file:  
    pickle.dump(topic_doc_idxes_dict, file)

index_file_2 = "index2.pkl"  

with open(index_file_2, 'wb') as file:  
    pickle.dump(doc_idx_topic_dict, file)

index_file_3 = "index3.pkl"  

with open(index_file_3, 'wb') as file:  
    pickle.dump(D, file)