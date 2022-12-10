# Disaster Response Pipeline Project (Udacity Data Science project)

### Project Motivation
In this project, disaster data from Appen is analyzed to build a model for an API that classifies disaster messages.
The dataset contains real messages that were sent during disaster events. A machine learning pipeline to categorize these events is created so that the messages can be sent to an appropriate disaster relief agency.
A web app is included, where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
Below are a few screenshots of the web app.

![image](https://user-images.githubusercontent.com/29317778/206826231-da02a791-e1a1-4faa-a6ae-cd9505dbb49a.png)
![image](https://user-images.githubusercontent.com/29317778/206826247-11c43d04-7231-4413-9776-a4a7c9faadc9.png)


### File Descriptions
Below are files for this project.
- app
    - master.html  # main page of web app
    - go.html  # classification result page of web app
    run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py # script to process data
    - InsertDatabaseName.db   # database to save clean data to

- models
    - train_classifier.py # script for a ML model
    - classifier.pkl  # saved model

- ETL Pipeline Preparation.ipynb # Jupyter notebook for ETL preparation
- ML Pipeline Preparation.ipynb # Jupyter notebook for ML pipeline preparation

- README.md


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

### Libraris used
The following Python libraries were used in this project.

pandas
sqlalchemy
nltk
sklearn
pickle
sys
