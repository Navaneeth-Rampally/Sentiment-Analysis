import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit_transformer(self, documents):        
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
            
        try:
            feature_matrix = self.vectorizer.fit_transform(documents)
            
            # --- FIX: Calculate this BEFORE using it in logger ---
            vocab_size = len(self.vectorizer.vocabulary_)            
            return feature_matrix
            
        except Exception as e:
            raise e

    def get_feature_names(self):
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()

if __name__ == "__main__":
    # --- FIX: Added commas here! ---
    sample_docs = [
        "This movie is too booring!.", 
        "Amazing movie to watch and i recommend this to everyone who like watching movies.",
        "Terrible movie.",
        "The lead actor playes an amazing role which makes it interesting to watch this movie."
    ]
    
    # Note: With only 4 documents, 'min_df=2' is strict. 
    # Only words appearing in 2+ sentences (like 'movie') will be kept.
    bow = FeatureExtractor(method='bow', max_features=50)
    bow_matrix = bow.fit_transformer(sample_docs)
    
    print("Features found:", bow.get_feature_names()[:10])