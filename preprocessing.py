import pandas as pd
import numpy as np
import re
import ssl
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import logging

# Writng logging configurations:
logger = logging.getLogger(__name__)

# Setting logger level
logger.setLevel(logging.DEBUG)

# formatting the logs
f = logging.Formatter('%(asctime)s-	%(levelname)s- 	%(message)s')

#Setting the file where the logs has to be stored automatically
fh = logging.FileHandler('preprocess.log')
fh.setFormatter(f)
logger.addHandler(fh)

## Downloading all required nltk essentials.
def download_nltk_data():
    """ Downloading Necessary nltk datasets."""
    logger.info("Prepairing to download Nltk essentials")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")

download_nltk_data()
logger.info('Downloaded nltk essentials sucessfully.')
class TextPreprocessor:
    def __init__(self, use_stemming = False, use_lemmatization = True):
        logger.info('Initializing TextPreprocessor..')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

    def step1_lowercase(self,text):
        logger.info('Converting text into lowercase')
        return text.lower()
    def step2_remove_html_tags(self,text):
        logger.info('removing html tags form the text')
        # Removing tags like <br>, <u>... form the text.
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    def step3_remove_url(self,text):
        logger.info('removing url..')
        #removing links like https://dbiwbd.com etc.... form the text
        clean_text = re.sub(r'http\S+|www\S+http\S+', '', text)
        return clean_text
    def step4_remove_punctuation(self,text):
        logger.info('removing punctuation')
        cleaned_text = text.translate(str.maketrans('','',string.punctuation))
        return cleaned_text
    def step5_remove_numbers(self,text):
        logger.info('removing numbers form the text')
        cleaned_text = re.sub(r'\d+', '',text)
        return cleaned_text
    def step6_remove_spaces(self,text):
        logger.info('removing unwanted and extra spaces from the text')
        cleaned_text = ' '.join(text.split())
        return cleaned_text
    def step7_tokenization(self,text):
        logger.info('conveting text into tokens')
        tokens = word_tokenize(text)
        return tokens
    def step8_remove_stopwords(self,tokens):
        logger.info('removing stopwords from tokens')
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens
    def step9_stemming(self,tokens):
        logger.info('stemming the tokens')
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return stemmed_tokens
    def step10_lemmatization(self,tokens):
        logger.info('lemmatizing the tokens')
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens
    def preprocess_text(self,text):
        logger.info('Now performing the preprocess pipeline step by step')
        text = self.step1_lowercase(text)
        text = self.step2_remove_html_tags(text)
        text = self.step3_remove_url(text)
        text = self.step4_remove_punctuation(text)
        text = self.step5_remove_numbers(text)
        text = self.step6_remove_spaces(text)
        tokens = self.step7_tokenization(text)
        tokens = self.step8_remove_stopwords(tokens)

        if self.use_stemming:
            tokens = self.step9_stemming(tokens)

        if self.use_lemmatization:
            tokens = self.step10_lemmatization(tokens)

        clean_text = ' '.join(tokens)
        logger.info('Text preprocessing successfully done. Now you can see the clean text.')
        return clean_text
    
if __name__ == "__main__":
    sample_text = """ <div>Hello NLP enthusiasts!!!  Check out this link: https://example.com/data for   more info... 
I cAn't believe the results (99% accuracy)!!! #MachineLearning @DataScience. 
The price is $50.00 & it's    very    cheap. Contact support@ai.net for help. <br> 
This is a    ReAlly   dirty  text sample.....  12345 </div>  """

preprocessor = TextPreprocessor(use_lemmatization=True)
cleaned = preprocessor.preprocess_text(sample_text)
print("\n --Preprocessed Text -- ")
print(cleaned)
