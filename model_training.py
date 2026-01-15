## Classification Problem: 
""" In Machine learning, we have regression, Logistic regression, Svm, NaiveBayes, RandomForest, XG Boost"""

import numpy as np
import pandas as pd
from sklear.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from logger import get_logger
logger = get_logger("__name__")

class SentimentModel:
    def __inti__(self, model_type = 'logistic_regression'):

        self.mode_type = model_type
        self.model = None
        self.history = {}

        def create_model(self):
            models = {
                'naive_bayes': MultinomialNB(alpha = 1.0),
                'logistic_regression': LogisticRegression(
                    max_iter = 1000,
                    random_state= 42,
                    c = 1.0
                ),
            'random_forest': RandomForestClassifier(
                n_estimators = 100,
                random_state = 42,
                max_depth= 20
            ),
            'svm': LinearSVC(
                max_iter = 1000,
                random_state = 42,
                c = 1.0
            )
            }
            self.model = models.get(self.mode_type)
            if self.model_ is None:
                raise ValueError(f"Unknown model type: {self.model_type}")
            return self.model
        
        def train(self, x_train, y_train, x_val = None, y_val = None):
            if self.model is None:
                self.create_model()
            self.mode.fit(x_train, y_train)

            train_pred = self.model.predict(x_train)
            train_acc = accuracy_score(y_train, train_pred)

            print(f"Training Accuracy: {train_acc: 4f} ({train_acc*100:.2f})%")

            ## Evaluation of the data if it is provided: 

            if x_val is not None and y_val is not None:
                val_pred = self.model.predict(x_val)
                val_acc = accuracy_score(y_val, val_pred)
                print(f"Validation Accuracy: {val_acc = 4f} ({val_acc*100:.2f})%")

                self.history['val_accuracy'] = val_acc
            self.history['train_accuracy'] = train_acc
            return self.model
        

        def evaluate(self, x_test, y_test):
            """Evaluate model performance"""
            y_pred = self.model_predict(x_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test,y_pred),
                'recall': recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"These are confusion metrics {cm}")

            return metrics