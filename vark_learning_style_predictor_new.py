import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 1: Load and Explore Data
print("Loading dataset...")
df = pd.read_csv('vark_learning_dataset.csv')

# Display basic info
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nLearning Style Distribution:")
print(df['Detected Learning Style (VARK)'].value_counts(normalize=True))

# Step 2: Preprocess Data
# Drop irrelevant columns
df = df.drop(['User ID', 'Recommended Learning Style'], axis=1)

# Define features and target
X = df.drop('Detected Learning Style (VARK)', axis=1)
y = df['Detected Learning Style (VARK)']

# Encode categorical variables
le_course = LabelEncoder()
le_topic = LabelEncoder()
le_difficulty = LabelEncoder()
le_learning_style = LabelEncoder()

X['Course'] = le_course.fit_transform(X['Course'])
X['Topic'] = le_topic.fit_transform(X['Topic'])
X['Difficulty Level'] = le_difficulty.fit_transform(X['Difficulty Level'])
y = le_learning_style.fit_transform(y)

# Scale numerical features
numerical_cols = ['Age', 'Video Time (mins)', 'Article Time (mins)', 
                  'Hands-on Time (mins)', 'Quiz Time (mins)', 
                  'Quiz Score (%)', 'Progress (%)', 'Attempts']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 4: Define and Train Models
print("\nTraining multiple models...")
models = {
    'SVM': SVC(kernel='rbf', C=1, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(multi_class='ovr', C=1, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
    'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# Dictionary to store accuracies and best model
model_accuracies = {}
best_model = None
best_accuracy = 0
best_model_name = ''

# Evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    
    # Classification report
    class_report = classification_report(y_test, y_pred, 
                                        target_names=le_learning_style.classes_, 
                                        zero_division=0, output_dict=True)
    macro_f1 = class_report['macro avg']['f1-score']
    
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=le_learning_style.classes_, 
                               zero_division=0))
    
    # Update best model based on accuracy (and macro F1 as tiebreaker)
    if accuracy > best_accuracy or (accuracy == best_accuracy and macro_f1 > model_accuracies.get('best_f1', 0)):
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        model_accuracies['best_f1'] = macro_f1

# Print model comparison
print("\nModel Comparison (Accuracy):")
for name, acc in model_accuracies.items():
    if name != 'best_f1':
        print(f"{name}: {acc:.4f}")
print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save model accuracies
with open('model_accuracies.pkl', 'wb') as f:
    pickle.dump(model_accuracies, f)

# Step 5: Save Best Model and Encoders
with open('vark_model_new.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler_new.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('le_course.pkl', 'wb') as f:
    pickle.dump(le_course, f)
with open('le_topic.pkl', 'wb') as f:
    pickle.dump(le_topic, f)
with open('le_difficulty.pkl', 'wb') as f:
    pickle.dump(le_difficulty, f)
with open('le_learning_style_new.pkl', 'wb') as f:
    pickle.dump(le_learning_style, f)

# Step 6: Prediction Function (Using Best Model)
def predict_learning_style(age, course, topic, difficulty_level, 
                          video_time, article_time, hands_on_time, 
                          quiz_time, quiz_score, progress, attempts):
    # Load saved model and encoders
    with open('vark_model_new.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_new.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_course.pkl', 'rb') as f:
        le_course = pickle.load(f)
    with open('le_topic.pkl', 'rb') as f:
        le_topic = pickle.load(f)
    with open('le_difficulty.pkl', 'rb') as f:
        le_difficulty = pickle.load(f)
    with open('le_learning_style_new.pkl', 'rb') as f:
        le_learning_style = pickle.load(f)
    
    # Prepare input data
    try:
        course_encoded = le_course.transform([course])[0]
        topic_encoded = le_topic.transform([topic])[0]
        difficulty_encoded = le_difficulty.transform([difficulty_level])[0]
    except ValueError as e:
        valid_courses = le_course.classes_
        valid_topics = le_topic.classes_
        valid_difficulties = le_difficulty.classes_
        return f"Error: Invalid input. Ensure course in {valid_courses}, " \
               f"topic in {valid_topics}, difficulty_level in {valid_difficulties}"
    
    # Create feature array
    features = np.array([[age, course_encoded, topic_encoded, difficulty_encoded,
                         video_time, article_time, hands_on_time, quiz_time,
                         quiz_score, progress, attempts]])
    
    # Scale numerical features
    features[:, [0, 4, 5, 6, 7, 8, 9, 10]] = scaler.transform(
        features[:, [0, 4, 5, 6, 7, 8, 9, 10]]
    )
    
    # Predict
    prediction = model.predict(features)[0]
    learning_style = le_learning_style.inverse_transform([prediction])[0]
    return learning_style

# Example Usage
sample_user = {
    'age': 21,
    'course': 'Machine Learning',
    'topic': 'OOP',
    'difficulty_level': 'Easy',
    'video_time': 13,
    'article_time': 3,
    'hands_on_time': 51,
    'quiz_time': 26,
    'quiz_score': 37,
    'progress': 33,
    'attempts': 1
}
result = predict_learning_style(**sample_user)
print(f"\nPredicted Learning Style for sample user (using {best_model_name}): {result}")