# Disaster Response Pipeline Project

## Objective
This project builds ML Classification Pipeline for Natural Language Processing problem. 
The input dataset consists of over 26k pre-processed Tweets translated to English concerning disaster and emergency 
messages posted on Twitter.

Objective of this project is to categorize each tweet with multiple descriptive labels of what kind of help is needed.

Example of a selected Tweet

```I live in Gonaives. I need help for the hospital in Raboto. We need water, food, and medication because we have a thousand people who need medical attention right now.```

## Project Strucutre




### Script usage instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

