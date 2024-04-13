import pickle
import numpy as np
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('index3.pkl', 'rb') as file:  
    D = pickle.load(file)

with open('index2.pkl', 'rb') as file:  
    doc_idx_topic_dict = pickle.load(file)

with open("skdocs.pkl", 'rb') as file:  
    sk_docs = pickle.load(file)

# Khởi tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()


# # Tiến hành chuyển đổi các tài liệu/văn bản về dạng các TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(sk_docs)

# Chuyển ma trận tfidf_matrix từ dạng cấu trúc thưa sang dạng đầy đủ để thuận tiện cho việc tính toán
tfidf_matrix = tfidf_matrix.todense()


def preprocess(doc):
    # Tiến hành xử lý các lỗi từ/câu, dấu câu, v.v. trong tiếng Việt với hàm text_normalize
    normalized_doc = text_normalize(doc)
    # Tiến hành tách từ
    tokens = word_tokenize(normalized_doc)
    # Tiến hành kết hợp các từ ghép trong tiếng Việt bằng '_'
    combined_tokens = [token.replace(' ', '_') for token in tokens]
    return (normalized_doc, combined_tokens)

def parse_query(query_text):
    (normalized_doc, combined_tokens) = preprocess(query_text)
    query_text = ' '.join(combined_tokens)
    query_tfidf_vector = vectorizer.transform([query_text])[0].todense()
    return query_tfidf_vector

# Viết hàm giúp tìm kiếm top-k (mặc định 10) các kết quả tài liệu/văn bản tương đồng với truy vấn 

def search(query_tfidf_vector, top_k=30):
    # Chúng ta sẽ thử tìm kiếm với top-10 kết quả
    search_results = {}
    for doc_idx, doc_tfidf_vector in enumerate(tfidf_matrix):
        # Tính mức độ tương đồng giữa truy vấn (q) và từng tài liệu/văn bản (doc_idx) bằng độ đo cosine
        cs_score = cosine_similarity(np.asarray(query_tfidf_vector), np.asarray(doc_tfidf_vector))
    #   cs_score = 1 - distance.cosine(query_tfidf_vector, doc_tfidf_vector)
        search_results[doc_idx] = cs_score[0,0]
    # Tiến hành sắp xếp các tài liệu/văn bản theo mức độ tương đồng từ cao -> thấp
    sorted_search_results = sorted(search_results.items(), key=lambda item: item[1], reverse=True)
    results = []
    # print('Top-[{}] tài liệu/văn bản có liên quan đến truy vấn.'.format(top_k))
    for idx, (doc_idx, sim_score) in enumerate(sorted_search_results[:top_k]):
        # print(' - [{}]. Tài liệu [{}], chủ đề: [{}] -> mức độ tương đồng: [{:.6f}]'.format(idx + 1, doc_idx, doc_idx_topic_dict[doc_idx], sim_score))
        results.append(doc_idx)
    return results


def rocchio(query_tfidf_vector, D_rel=[1, 2, 3]):
    # Xác định danh sách các tài liệu/văn bản có liên quan (D_rel) và không liên quan (D_irel) đến truy vấn (q)
    # ----------------------------------------------- #
    # D_rel = [D.index(d) for d in D if d[0] == 'Y tế & sức khỏe']
    D_rel = [eval(i) for i in D_rel]
    print(D_rel)
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
    return search(query_tfidf_vector_QR)
