import pandas as pd
#importing datasets from Huggingface
from datasets import load_dataset   
import logging

logger = logging.getLogger(__name__)
#setting logger level to DEBUG
logger.setLevel(logging.DEBUG)
#formatting the logger
format = logging.Formatter('%(asctime)s-	%(levelname)s- 	%(message)s')

#Creating the new file for storing the logs
fh = logging.FileHandler('ingestion.log')
fh.setFormatter(format)
logger.addHandler(fh)

def load_imdb_data(sample_size = None):
    """" Loading IMDB Movie dataset from huggingface performing and learning sentiment analysis."""
    logger.info('Loading the required dataset..')
    ## Load datset 
    dataset = load_dataset('imdb')
    logger.info('Dataset loaded successfully.')

    # Converting into pandas dataframes train and test respectively
    logger.info('Converting into dataframes')
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Creating a new column named sentiment and mapping them into negative and positive.
    logger.info('Creating a new coloumn named sentiment')
    train_df['sentiment'] = train_df['label'].map({0:'Negative',1:'positive'})
    test_df['sentiment'] = train_df['label'].map({0:'Negative',1:'positive'})

    #renaming the columns 
    logger.info('Renaming the columns')
    train_df = train_df.rename(columns = {'text':'review'})
    test_df = test_df.rename(columns={'text':'review'})

    #fetching sample data based on size requested by user.
    logger.info('Fetching the sample database which is required by the user')
    train_df = train_df.sample(n=min(sample_size,len(train_df)),random_state=42)
    test_df = test_df.sample(n=min(sample_size,len(test_df)),random_state=42)

    print(f" Loaded {len(train_df)} training sample.")
    print(f" Loaded {len(test_df)} testing sample.")

    return train_df, test_df
logger.info('Executing the main code..')
if __name__ == "__main__":
    train_df, test_df = load_imdb_data(sample_size=1000)
    print(train_df.head())
    print(test_df.head())
