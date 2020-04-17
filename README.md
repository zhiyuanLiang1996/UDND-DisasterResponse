# Disaster Response Pipeline Project

### Summary
In this project, I set out to categorize messages from different sources during disasters. I cleaned the data, use NLP to process the  raw message and trained a Random Forest classifier. Finally, I used this classifier to classify messages and deployed it to the web. This project is useful because during disasters, message spur in numbers and it's critical to designate them to corresponding department.

### Files introduction
- 'data' folder contains ETL pipeline.
- 'model' folder contains ML pipeline.
- 'app' folder contains Web deployment related files. Sample html files are also included

### Instructions for running:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
