import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load_data
    This function loads and merge datasets to a single pandas dataframe.
    
    Inputs:
        messages_filepath: messages filepath
        categories_filepath: categories filepath
    
    Returns:
        df: merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='left')
    
    return df


def clean_data(df):
    """
    clean_data
    This function is for cleaning data before being pushed to a database
    
    Inputs:
        df: merged dataframe
    Return:
        df: merged dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    df['related'].replace(2,1,inplace=True)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    save_data
    This function saves the clean dataset into a sqlite database.
    
    Inputs:
        df: cleaned dataframe
        database_filename: database filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
