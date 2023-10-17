
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from joblib import load


df = pd.read_csv("Project 1 Data.csv")

X = df['X']
Y = df['Y']
Z = df['Z']
step = df['Step']



#Prevent data breakage by dropping all n/a values
df = df.dropna()

my_scaler = StandardScaler()
my_scaler.fit(df)

#Step 2 Plotting the 3D graph and perfomre statistical analysis
print("")
print("Step 2, Data Visualization")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#To give color variation based on step size
ax.scatter(X, Y, Z, c=step, cmap='viridis')
#Labeling the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#Show the plot
plt.show()
#Analyzing and printing summary statistics
print(df.describe())

#Step 3 - Correlation Analysis

correlation_matrix = df.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix', fontsize=20)
plt.show()

#Step 4 - Classification Model Development/Engineering


#Defining data sets (test vs training)
x = df[['X','Y','Z']]
Y = df[['Step']]

#Scaling and Splitting data
xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size=0.2,random_state=15)
sc = StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)

#Function to fit and grid search a model
def get_best_model_and_accuracy(model, params, xtrain, ytrain, xtest, ytest):
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)  # Instantiating GridSearchCV
    grid.fit(xtrain, ytrain)  # Fitting the model and the parameters to the GridSearchCV
    print(f"Best Accuracy: {grid.best_score_}")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Accuracy on test data: {grid.score(xtest, ytest)}")

# Define the models and parameters
models_and_parameters = {
    "logistic_regression": (LogisticRegression(max_iter=5000), 
                            {'C': [0.001, 0.01, 0.1, 1, 10]}),
    "svm": (SVC(), 
            {'C': [0.001, 0.01, 0.1, 1, 10]}),
    "random_forest": (RandomForestClassifier(), 
                      {'n_estimators': [10, 50, 100]})
}

# Train the models and display the best parameters
for model_name, (model, params) in models_and_parameters.items():
    print(f"==== Starting Grid Search for {model_name} ====")
    get_best_model_and_accuracy(model, params, xtrain, ytrain, xtest, ytest)
    
#Step 5 - Model Performance Analysis

# Function to display the confusion matrix
def plot_confusion_matrix(ytrue, ypred, labels):
    cm = confusion_matrix(ytrue, ypred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Define the evaluation metrics
evaluation_metrics = {
    "f1_score": f1_score,
    "precision": precision_score,
    "accuracy": accuracy_score
}

# Compare the overall performance of each model
for model_name, (model, params) in models_and_parameters.items():
    model.fit(xtrain, ytrain)  # Fit the model on the entire training data
    ypred = model.predict(xtest)  # Make predictions on the test data
    print(f"=== Evaluation Metrics for {model_name} ===")
    for metric_name, metric_func in evaluation_metrics.items():
        print(f"{metric_name.capitalize()}: {metric_func(ytest, y_pred, average='weighted')}")
    # Creating a confusion matrix
    unique_labels = sorted(ytest['Step'].unique())
    plot_confusion_matrix(ytest, ypred, labels=unique_labels)
    
    
#Step 6 - Model Evaluation

# Save the selected model
selected_model = models_and_parameters['svm'][0]  # Selecting the SVM model for demonstration
jl.dump(selected_model, 'selected_model.joblib')

# Load the model
loaded_model = jl.load('selected_model.joblib')

# Reshape the new sample data
new_sample_data = np.array([9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3])
reshaped_data = new_sample_data.reshape(-1, 3)  # Reshape the data to have 3 features

# Predict the maintenance step for the new sample data
predicted_steps = loaded_model.predict(reshaped_data)

# Print the predicted maintenance steps
print("Predicted Maintenance Steps:")
for i, step in enumerate(predicted_steps):
    print(f"Data point {i + 1}: {new_sample_data[i]} --> Predicted Step: {step}")