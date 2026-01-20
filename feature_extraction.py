import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureExtractor:
    """ Feature extraction methods using BOW and TF-IDF"""

    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit_transform(self, documents):       
        """ Fit and transform document to feature vectors""" 
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
            feature_matrix = self.vectorizer.fit_transform(documents)
            
            # --- FIX: Calculate this BEFORE using it in logger ---
            vocab_size = len(self.vectorizer.vocabulary_)            
            return feature_matrix
    def transform_(self, documents):
        """ Transform new documents using the appropriate vectorizer"""
        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been fitted yet.")
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        """ Get feature names from the vectorizer"""
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
    bow_matrix = bow.fit_transform(sample_docs)
    print(bow.get_feature_names()[:10])

    tfidf = FeatureExtractor(method='tfidf', max_features=50)
    tfidf_matrix = tfidf.fit_transform(sample_docs)
    print(tfidf.get_feature_names()[:10])
    
