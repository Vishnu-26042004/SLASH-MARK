import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def read_documents(file_paths):
    documents = []
    for path in file_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                documents.append(file.read().strip())
        else:
            print(f"Warning: File {path} not found.")
            documents.append("")
    return documents

def detect_plagiarism(docs, threshold=0.5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("Plagiarism Report:")
    plagiarism_found = False
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            print(f"Document {i+1} and Document {j+1} similarity: {similarity_matrix[i, j]:.2f}")
            if similarity_matrix[i, j] > threshold:
                print(f"⚠️ Potential Plagiarism: Document {i+1} and Document {j+1} have a similarity of {similarity_matrix[i, j]:.2f}")
                plagiarism_found = True
    
    if not plagiarism_found:
        print("No significant plagiarism detected.")

document_files = [
    'C:/Users/user/Desktop/Slashmark/document1.txt', 
    'C:/Users/user/Desktop/Slashmark/document2.txt'
]

documents = read_documents(document_files)

detect_plagiarism(documents, threshold=0.3)
