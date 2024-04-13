import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

# Chọn danh sách các chủ đề của tài liệu/văn bản cho thử nghiệm
DIR_PATH = 'dataset/'
topics = sorted(os.listdir(DIR_PATH))

# Tạo một tập dữ liệu thử nghiệm gồm các tài liệu/văn bản thuộc về 2-3 chủ đề
# Cấu trúc dữ liệu dạng list - lưu thông tin danh sách các tài liệu/văn bản thuộc chủ đề khác nhau
# Mỗi tài liệu/văn bản sẽ tổ chức dạng 1 tuple với: (topic, nội_dung_văn_bản, danh_sách_token)
D = []

# Viết hàm tiền xử lý và tách từ tiếng Việt
def preprocess(doc):
  # Tiến hành xử lý các lỗi từ/câu, dấu câu, v.v. trong tiếng Việt với hàm text_normalize
  normalized_doc = text_normalize(doc)
  # Tiến hành tách từ
  tokens = word_tokenize(normalized_doc)
  # Tiến hành kết hợp các từ ghép trong tiếng Việt bằng '_'
  combined_tokens = [token.replace(' ', '_') for token in tokens]
  return (normalized_doc, combined_tokens)

# Viết hàm lấy danh sách các văn bản/tài liệu thuộc các chủ đề khác nhau
def fetch_doc_by_topic(topic):
  data_root_dir_path = DIR_PATH + topic
  docs = []
  for file_name in os.listdir(data_root_dir_path):
    file_path = os.path.join(data_root_dir_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
      lines = []
      for line in f:
        line = line.lower().strip()
        lines.append(line)
    doc = " ".join(lines)
    clean_doc = re.sub('\W+',' ', doc)
    (normalized_doc, tokens) = preprocess(clean_doc)
    docs.append((topic, normalized_doc, tokens))
  return docs

# Cấu trúc dữ liệu dictionary để lưu thông tin chủ đề-tài liệu, nhằm hỗ trợ cho việc tìm kiếm nhanh
topic_doc_idxes_dict = {}
doc_idx_topic_dict = {}

# Duyệt qua từng chủ đề
doc_idx = 0
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
  
# Khởi tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()

# Chúng ta sẽ tạo ra một tập danh sách các tài liệu/văn bản dạng list đơn giản để thư viện Scikit-Learn có thể đọc được
sk_docs = []

# Duyệt qua từng tài liệu/văn bản có trong (D)
for (topic, normalized_doc, tokens) in D:
  # Chúng ta sẽ nối các từ/tokens đã được tách để làm thành một văn bản hoàn chỉnh
  text = ' '.join(tokens)
  sk_docs.append(text)

# Tiến hành chuyển đổi các tài liệu/văn bản về dạng các TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(sk_docs)

# Chuyển ma trận tfidf_matrix từ dạng cấu trúc thưa sang dạng đầy đủ để thuận tiện cho việc tính toán
tfidf_matrix = tfidf_matrix.todense()

# Viết hàm giúp chuyển đổi truy vấn dạng text sang tfidf vector
def parse_query(query_text):
  (normalized_doc, combined_tokens) = preprocess(query_text)
  query_text = ' '.join(combined_tokens)
  query_tfidf_vector = vectorizer.transform([query_text])[0].todense()
  return query_tfidf_vector

# Viết hàm giúp tìm kiếm top-k (mặc định 10) các kết quả tài liệu/văn bản tương đồng với truy vấn 
from sklearn.metrics.pairwise import cosine_similarity
def search(query_tfidf_vector, top_k = 10):
  search_results = {}
  for doc_idx, doc_tfidf_vector in enumerate(tfidf_matrix):
      # Tính mức độ tương đồng giữa truy vấn (q) và từng tài liệu/văn bản (doc_idx) bằng độ đo cosine
      cs_score = cosine_similarity(np.asarray(query_tfidf_vector), np.asarray(doc_tfidf_vector))
    #   cs_score = 1 - distance.cosine(query_tfidf_vector, doc_tfidf_vector)
      search_results[doc_idx] = cs_score[0,0]
  # Tiến hành sắp xếp các tài liệu/văn bản theo mức độ tương đồng từ cao -> thấp
  sorted_search_results = sorted(search_results.items(), key=lambda item: item[1], reverse=True)
  print('Top-[{}] tài liệu/văn bản có liên quan đến truy vấn.'.format(top_k))
  for idx, (doc_idx, sim_score) in enumerate(sorted_search_results[:top_k]):
    print(' - [{}]. Tài liệu [{}], chủ đề: [{}] -> mức độ tương đồng: [{:.6f}]'.format(idx + 1, doc_idx, doc_idx_topic_dict[doc_idx], sim_score))



# Thử một truy vấn với chủ đề [the-thao]
query_text = 'bác sĩ'

# Chuyển đổi truy vấn về dạng vector tfidf
query_tfidf_vector = parse_query(query_text)

# Chúng ta sẽ thử tìm kiếm với top-10 kết quả
top_k = 10

  # Tiến hành tìm kiếm thử trong tập dữ liệu với truy vấn
  # return search(query_tfidf_vector, top_k)
search(query_tfidf_vector, top_k)



# Xác định danh sách các tài liệu/văn bản có liên quan (D_rel) và không liên quan (D_irel) đến truy vấn (q)
# D_rel = [27, 32, 0, 33, 24, 19, 7, 5]
# D_irel = [108, 101]

# # ----------------------------------------------- #
D_rel = [D.index(d) for d in D if d[0] == 'Y tế & sức khỏe']
D_irel = [D.index(d) for d in D if D.index(d) not in D_rel]

# Tạo mới một query mở rộng [query_tfidf_vector_QR] cho [query_tfidf_vector]
query_tfidf_vector_QR = query_tfidf_vector

# Chúng ta định nghĩa các hằng số \apha, \beta và \gamma
alpha = 0.0
beta = 1.0
gamma = 1.0

# Tiến hành xác định tổng bình thường hóa (normalized sum) các vectors của các tài liệu trong tập (D_rel)
sum_normalized_rel = np.zeros(query_tfidf_vector_QR.shape)
for doc_idx in D_rel:
  doc_tfidf_vector_rel = tfidf_matrix[doc_idx]
  sum_normalized_rel += doc_tfidf_vector_rel

# Quá trình bình thường hóa vector [sum_normalized_rel]
sum_normalized_rel = (beta / len(D_rel)) * sum_normalized_rel

# Tiến hành xác định tổng bình thường hóa (normalized sum) các vectors của các tài liệu trong tập (D_irel)
sum_normalized_irel = np.zeros(query_tfidf_vector_QR.shape)
for doc_idx in D_irel:
  doc_tfidf_vector_irel = tfidf_matrix[doc_idx]
  sum_normalized_irel += doc_tfidf_vector_irel

# Quá trình bình thường hóa vector [sum_normalized_irel]
sum_normalized_irel = (gamma / len(D_irel)) * sum_normalized_irel

query_tfidf_vector_QR = (alpha * query_tfidf_vector) + sum_normalized_rel - sum_normalized_irel

# Tiến hành tìm kiếm lại với truy vấn mới[query_tfidf_vector_QR]
search(query_tfidf_vector_QR, top_k)

