import pandas as pd

# Load the dataset
file_path = r"/content/Mac ML - Clean v7 w weather.xlsx"
data = pd.read_excel(file_path)

from datetime import timedelta

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Initialize columns for features
data['Total Classes Next Day'] = 0
data['Liv Football Practice Next Day'] = 0

# Iterate through the dataset to calculate features for each day
for i in range(len(data) - 1):
    # Total number of classes next day
    total_classes_next_day = 0
    for j in range(1, 5):  # Assuming up to 4 classes per day
        if pd.notnull(data.iloc[i + 1][f'Class {j} Code']):
            total_classes_next_day += 1
    data.at[i, 'Total Classes Next Day'] = total_classes_next_day

    # Liv's football practice next day
    if pd.notnull(data.iloc[i + 1]['Liv\'s football practice schedule']):
        data.at[i, 'Liv Football Practice Next Day'] = 1

# Display the modified dataset with the new features
data[['Date', 'Total Classes Next Day', 'Liv Football Practice Next Day']].head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
from datetime import datetime

# Selecting target variable
y = data['Mac going out level']

# Selecting categorical columns that need to be one-hot encoded
categorical_cols = [col for col in data.columns if 'Code' in col]

# Selecting numerical columns to be scaled
numerical_cols = ['Scale liv would go out','Total Classes Next Day', 'Liv Football Practice Next Day']

# Define columns that are time but not in numerical format yet, will convert to hour of day
time_cols = [col for col in data.columns if 'Start' in col or 'End' in col or 'schedule' in col]

# Map 'Night of the week' to corresponding influence levels
night_influence_mapping = {
    "N/A": 1, "DBs_T": 6, "Stages": 3, "House/Ale": 5, "DBs_S": 7, "Brass": 9
}
data['Night of the week'] = data['Night of the week'].map(night_influence_mapping)

# Initialize new feature columns
data['Earliest Mac Class Next Day'] = 24  # Assuming 24 as a placeholder for no class
data['Earliest Liv Class Next Day'] = 24  # Same as above
data['Assignment Influence'] = 0  # Placeholder for assignment influence score

# Calculating Mac and Liv earliest class time and the assignment weightings
for i in range(len(data) - 1):
    # Calculate the earliest class for Mac next day
    mac_class_times_next_day = [pd.to_datetime(data.iloc[i + 1][f'Class {j} Start'], format='%H:%M:%S', errors='coerce') for j in range(1, 5) if pd.notnull(data.iloc[i + 1][f'Class {j} Start'])]
    if mac_class_times_next_day:
        # Filter out NaT values resulted from 'coerce' in to_datetime
        mac_class_times_next_day = [time for time in mac_class_times_next_day if time is not pd.NaT]
        if mac_class_times_next_day:  # Check again in case all were NaT
            earliest_time = min(mac_class_times_next_day)
            data.at[i, 'Earliest Mac Class Next Day'] = earliest_time.hour + earliest_time.minute / 60

    # Calculate the earliest class for Liv next day
    liv_class_times_next_day = [pd.to_datetime(data.iloc[i + 1][f'Liv Class {j} Start'], format='%H:%M:%S', errors='coerce') for j in range(1, 5) if pd.notnull(data.iloc[i + 1][f'Liv Class {j} Start'])]
    if liv_class_times_next_day:
        # Filter out NaT values resulted from 'coerce' in to_datetime
        liv_class_times_next_day = [time for time in liv_class_times_next_day if time is not pd.NaT]
        if liv_class_times_next_day:  # Check again in case all were NaT
            earliest_time = min(liv_class_times_next_day)
            data.at[i, 'Earliest Liv Class Next Day'] = earliest_time.hour + earliest_time.minute / 60


    # Calculate assignment influence by assignment weighting and due date proximity
    if pd.notnull(data.iloc[i]['Date']) and pd.notnull(data.iloc[i]['Assignment Weighting']):
        due_date = pd.to_datetime(data.iloc[i]['Assignment Due Date'])
        days_until_due = (due_date - data.iloc[i]['Date']).days
        if days_until_due >= 0 and days_until_due <= 2:  # If due today or in the next 2 days
            influence = data.iloc[i]['Assignment Weighting'] / (days_until_due + 1)  # Calculating assignment influence
            data.at[i, 'Assignment Influence'] = influence

# Updating numberical and categorial columns
numerical_cols_updated = numerical_cols + ['Temp', 'Feels Like', 'Precip', 'Humidity', 'Night of the week', 'Cost of going out on the night', 'Earliest Mac Class Next Day', 'Earliest Liv Class Next Day', 'Assignment Influence']
categorical_cols += ['Icon']

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Preprocessing for time data
def extract_hour(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.hour.fillna(0)
    return df

# Apply time transformation
data = extract_hour(data, time_cols)

# Preprocessor with the new numerical_cols_updated
preprocessor_updated = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_updated),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define a model comparison function
def compare_models(models, X, y):
    model_names = []
    cv_mae_scores = []
    cv_rmse_scores = []

    # Perform cross-validation for each model
    for model in models:
        # Create a pipeline with the preprocessor and the current model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_updated),
                                   ('model', model)])

        # Calculate MAE scores using cross-validation
        scores_mae = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae_scores.append(-scores_mae.mean())

        # Calculate RMSE scores using cross-validation
        scores_rmse = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse_scores.append(-scores_rmse.mean())

        # Save the model name for the results
        model_names.append(model.__class__.__name__)

    # Combine the results into a DataFrame for easy viewing
    return pd.DataFrame({'Model': model_names, 'MAE': cv_mae_scores, 'RMSE': cv_rmse_scores})

# Initialize a list of models to compare
models_to_compare = [
    LinearRegression(),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(n_jobs=-1, random_state=42),
    GradientBoostingRegressor(random_state=42),
    SVR(),
    Ridge(random_state=42),
    Lasso(random_state=42)
]

# Run the model comparison
comparison_results = compare_models(models_to_compare, data, y)

comparison_results

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = GradientBoostingRegressor(random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Prepare data for modeling
X = data.drop('Mac going out level', axis=1)
y = data['Mac going out level']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Saving the trained pipeline
model_filename = 'Mac v6 model GBR.joblib'
joblib.dump(pipeline, model_filename)

from google.colab import files
files.download(model_filename)

