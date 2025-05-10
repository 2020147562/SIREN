# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# model = SentenceTransformer("all-MiniLM-L6-v2")
# db = ["예시대화1", "예시대화2", "예시대화3"]
# vecs = model.encode(db)
# index = faiss.IndexFlatL2(vecs.shape[1])
# index.add(vecs)

# def search_similar_cases(query: str):
#     q_vec = model.encode([query])
#     D, I = index.search(q_vec, k=3)
#     return [db[i] for i in I[0]]
