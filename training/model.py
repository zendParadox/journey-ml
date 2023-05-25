import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Baca dataset
data = pd.read_csv('datasets/dataset_journey.csv')

# print(data.head())

# Menggabungkan fitur-fitur yang relevan menjadi satu teks
data['text'] = data['disability_type'] + ' ' + data['skills_one'] + ' ' + data['skills_two'] + ' ' + data['company'] + ' ' + data['city'] + ' ' + data['province'] + ' ' + data['job_type'] + ' ' + data['position']

# Menghitung representasi vektor TF-IDF untuk setiap dokumen teks
tfidf = TfidfVectorizer()
data['text'].fillna('', inplace=True)
tfidf_matrix = tfidf.fit_transform(data['text'])

# Menghitung similarity score menggunakan cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Membangun fungsi untuk merekomendasikan pekerjaan berdasarkan judul
def recommend_jobs(title, cosine_similarities, data, top_n=5):
    # Mengambil index pekerjaan berdasarkan judul
    idx = data[data['position'] == title].index[0]
    
    # Mengambil similarity scores pekerjaan yang serupa
    similarity_scores = list(enumerate(cosine_similarities[idx]))
    
    # Mengurutkan pekerjaan berdasarkan similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Mengambil top N pekerjaan yang paling serupa
    top_jobs = [data['position'].iloc[i[0]] for i in similarity_scores[1:top_n+1]]
    
    return top_jobs

# Contoh penggunaan: merekomendasikan pekerjaan berdasarkan judul "Data Analyst"
recommended_jobs = recommend_jobs("Data Analyst", cosine_similarities, data)
print(recommended_jobs)
