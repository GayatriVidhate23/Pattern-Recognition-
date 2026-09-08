import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import PyPDF2
import docx

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

def extract_text(file_content: bytes, filename: str) -> str:
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif filename_lower.endswith('.txt'):
        return file_content.decode('utf-8', errors='ignore')
    else:
        return ""

def clean_text(text: str) -> str:
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_top_keywords(tfidf_matrix, vectorizer, cluster_labels, n_clusters: int):
    feature_names = vectorizer.get_feature_names_out()
    keywords = {}
    for i in range(n_clusters):
        # get docs for this cluster
        cluster_docs = tfidf_matrix[cluster_labels == i]
        if cluster_docs.shape[0] == 0:
            keywords[i] = []
            continue
        avg_scores = np.asarray(cluster_docs.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-5:][::-1]
        keywords[i] = [feature_names[idx] for idx in top_indices]
    return keywords

def cluster_documents(documents):
    # documents: list of dicts {'id': str, 'filename': str, 'text': str}
    texts = [doc['text'] for doc in documents]
    
    if len(texts) < 2:
        raise ValueError("At least 2 documents are required for clustering.")
        
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Ensure distance matrix is symmetric and has non-negative values
    distance_matrix = np.maximum(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Compute linkage for dendrogram
    # Using 'ward' linkage which is default and typically gives good results
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    # Determine n_clusters dynamically based on number of documents
    # A simple heuristic: sqrt(N) or N/2
    n_clusters = max(2, int(np.sqrt(len(texts))))
    if n_clusters >= len(texts):
        n_clusters = max(1, len(texts) - 1)
        
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='ward'
    )
    # Fit clustering on the actual Euclidean distances of TF-IDF vectors (which correlates with cosine distance for L2 normalized vectors)
    cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
    
    # Extract keywords
    num_actual_clusters = len(set(cluster_labels))
    cluster_keywords = get_top_keywords(tfidf_matrix, vectorizer, cluster_labels, num_actual_clusters)
    
    # 2D projection using PCA
    pca = PCA(n_components=2)
    if len(texts) >= 2:
        points_2d = pca.fit_transform(tfidf_matrix.toarray())
    else:
        points_2d = np.zeros((len(texts), 2))
        
    return {
        'cluster_labels': cluster_labels.tolist(),
        'cluster_keywords': cluster_keywords,
        'similarity_matrix': similarity_matrix.tolist(),
        'linkage_matrix': Z.tolist(),
        'points_2d': points_2d.tolist(),
        'tfidf_matrix': tfidf_matrix
    }
