import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt


file_path = '/mnt/data/Dataset .csv'
df = pd.read_csv(file_path)


df.head()


df = df[['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating', 'Locality', 'City', 'Latitude', 'Longitude']].dropna()

map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], 
                  popup=row['Restaurant Name'] + ' - ' + row['Cuisines']).add_to(restaurant_map)

restaurant_map.save("restaurant_map.html")

grouped_by_locality = df.groupby('Locality').size().reset_index(name='Restaurant Count')
grouped_by_city = df.groupby('City').size().reset_index(name='Restaurant Count')


locality_stats = df.groupby('Locality').agg({
    'Aggregate rating': 'mean',
    'Price range': 'mean'
}).reset_index()

city_stats = df.groupby('City').agg({
    'Aggregate rating': 'mean',
    'Price range': 'mean'
}).reset_index()

top_localities = grouped_by_locality.sort_values(by='Restaurant Count', ascending=False).head(10)
top_cities = grouped_by_city.sort_values(by='Restaurant Count', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_localities['Locality'], top_localities['Restaurant Count'], color='skyblue')
plt.xlabel('Number of Restaurants')
plt.title('Top 10 Localities by Number of Restaurants')
plt.gca().invert_yaxis()
plt.show()


plt.figure(figsize=(10, 6))
plt.barh(top_cities['City'], top_cities['Restaurant Count'], color='lightgreen')
plt.xlabel('Number of Restaurants')
plt.title('Top 10 Cities by Number of Restaurants')
plt.gca().invert_yaxis()
plt.show()

print("Locality Statistics:\n", locality_stats.head())
print("City Statistics:\n", city_stats.head())
