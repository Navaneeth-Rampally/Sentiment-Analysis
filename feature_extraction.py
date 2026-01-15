import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class FeatureExtractor:
    """ Implementing feature extractor methods using BOW and TF-IDF"""

    def __inti__(self, method = 'tfidf', max_features = 5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit_transformer(self, documents):
        """ Fit and transform document to feature vectors."""

        if self.methos == 'bow':
            self.vectorizer = CountVectorizer(
                max_features= self.max_features,
                min_df= 2,
                max_df= 0.8
            )
        else: 
            self.vectorizer = TfidfVectorizer(
                max_features = self.max_features,
                min_df = 2
                max_df = 0.8
            )
        feature_matrix = self.vectorizer.fit_transfor(documents)

        print(feature_matrix.shape)
        print(len(self.vectorizer.vocabulary_))
        return feature_matrix
    def transform(self, documents):
        """ Transform new documents using the appropriate vectorizer"""

        if self.vectorizer is None:
            raise ValueError("vectorizer not fitted yet, call the fit_transform initially")
        return self.vectorizer.transform(documents)
    def get_feature_names(self):
        """ Get Feature names."""

        if self.vectorizer is None:
            return[]
        return self.vectorizer.get_feature_names_out()
    
    if __name__ == "__main__":
        sample_docs == [
            "This movie is too booring!."
            "Amazing movie to watch and i recommend this to everyone who like watching movies."
            "Terrible movie."
            "The lead actor playes an amazing role which makes it interesting to watch this movie."

        ]
        bow = FeatureExtractor(method = 'bow', max_features = 50)
        bow_matrix = bow.fit_transform(sample_docs)
        print(tfidf.get_feature_names()[:10])