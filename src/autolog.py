import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os
import joblib

dagshub.init(repo_owner='shekharsharma100001', repo_name='MLflow-exp-tracking', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/shekharsharma100001/MLflow-exp-tracking.mlflow')


#load wine dataset
wine = load_wine()
x = wine.data
y = wine.target

# train test split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10, random_state=42)

#define the params for model
max_depth = 5
n_estimators = 10


#autolog
mlflow.autolog()
# if we do not create any experiment, a all the runs are saved in default experiment
# creating new experiment
mlflow.set_experiment('new-exp1')
# with mlflow.start_run(experiment_id=767329552007102233): we can also use this to set new experiment
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth)        no need to log them explicitly, it will automatically be logged
    # mlflow.log_param('n_estimators', n_estimators)

    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot = True, fmt='d', cmap='Blues', xticklabels= wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    # save plot
    plt.savefig('confusion-matrix.png')

    # log artifacts using mlflow
    mlflow.log_artifact(__file__)  # file should be loogged explicitly, autolog will not log it
    # mlflow.log_artifact('confusion-matrix.png')
    # tags
    mlflow.set_tags({'Author': 'Shekhar', 'Project': 'Wine classification'})

    # Log the model
    # mlflow.sklearn.log_model(rf, "Random Forest Model")

    # Save the model locally
    os.makedirs("model", exist_ok=True)
    joblib.dump(rf, "model/model.pkl")

    # Log the model manually
    # mlflow.log_artifact("model/model.pkl", artifact_path="model") model will also be automatically logged


    print(accuracy)