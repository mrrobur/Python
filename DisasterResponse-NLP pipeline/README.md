# Disaster Response Pipeline Project

## Objective
This project builds ML Classification Pipeline for Natural Language Processing problem. 
The input dataset consists of over 26k pre-processed Tweets translated to English concerning disaster and emergency 
messages posted on Twitter.

Objective of this project is to categorize large number of tweets with multiple descriptive labels of what kind of help is needed.

## Summary
Final model has been deployed using Random Forest Classifier after prior text preparation and TF-IDF transformer.
The best cross-validated model is achieving average 0.612 f1-score among all classes spanning from 0.481 to 0.923.

GridSearch tested with different n-grams confirmed that the best score in this case is with using 1-grams.

#### Possible improvement
It is worth investigating the performance of the model using Word Embeddings in place of TF-IDF approach.

## Project File Strucutre
```
.
├── README.md
├── app                                     // Flask Web App
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data                                    // Data for preprocessing and modelling
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── development                             // Jupyter Notebooks for development stage
│   ├── ETL Pipeline Preparation.ipynb
│   └── ML Pipeline Preparation.ipynb
├── models                                  // ML model train script and pickled model
│   ├── classifier.pkl
│   └── train_classifier.py
└── struct

5 directories, 13 files

```

### Script usage instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  
2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/