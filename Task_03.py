import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = 'Dataset .csv'
df = pd.read_csv(file_path)

df = df[['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating']].dropna()


label_encoder = LabelEncoder()
df['Cuisines'] = label_encoder.fit_transform(df['Cuisines'])


X = df[['Price range', 'Aggregate rating']]
y = df['Cuisines']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Aggregate rating', 'Price range'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)


performance_analysis = pd.DataFrame(classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)).transpose()
performance_analysis


cuisine_counts = df['Cuisines'].value_counts()
cuisine_counts
