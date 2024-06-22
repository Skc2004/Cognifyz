
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('Dataset.csv')


print(df.head())


df.fillna(df.mean(), inplace=True)
df['Cuisines'].fillna(df['Cuisines'].mode()[0], inplace=True)
df['Price Range'].fillna(df['Price Range'].mode()[0], inplace=True)

# Encoding categorical variables
categorical_features = ['Cuisines', 'Price Range']
numerical_features = ['rating'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


X = preprocessor.fit_transform(df)

user_preference = {
    'cuisine': 'Italian',
    'price_range': '$$',
    'rating': 4.5
}


user_pref_df = pd.DataFrame([user_preference])


user_pref_transformed = preprocessor.transform(user_pref_df)

similarity_scores = cosine_similarity(user_pref_transformed, X)


top_indices = similarity_scores[0].argsort()[-5:][::-1]


recommended_restaurants = df.iloc[top_indices]
print("Recommended Restaurants:")
print(recommended_restaurants)


def recommend_restaurants(user_pref, df, preprocessor, top_n=5):
    user_pref_df = pd.DataFrame([user_pref])
    user_pref_transformed = preprocessor.transform(user_pref_df)
    similarity_scores = cosine_similarity(user_pref_transformed, X)
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]


sample_user_pref = {
    'cuisine': 'Mexican',
    'price_range': '$',
    'rating': 4.0
}
recommendations = recommend_restaurants(sample_user_pref, df, preprocessor, top_n=5)
print("Sample User Preferences Recommendations:")
print(recommendations)
