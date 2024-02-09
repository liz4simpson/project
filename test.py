import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib

# Загрузка данных
data = pd.read_csv("test.csv")  # Замените "your_dataset.csv" на путь к вашему файлу с данными

# Вывод первых нескольких строк данных для проверки
print(data.head())

# Предварительный анализ данных
print(data.info())
print(data.describe())

# Выбор признаков для кластеризации (можно использовать цены открытия, закрытия, максимальные и минимальные цены)
X = data[['Open', 'Close', 'High', 'Low']]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

## Создание модели кластеризации (например, k-means с 3 кластерами)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Добавление меток кластеров к исходным данным
data['Cluster'] = kmeans.labels_

# Визуализация кластеров
plt.scatter(data['Date'], data['Close'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Clusters of Stock Prices')
plt.show()

# Вывод средних значений цен для каждого кластера
print(data.groupby('Cluster')[['Open', 'High', 'Low', 'Close', 'Volume']].mean())

# Вычисление индекса силуэта
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("Средний индекс силуэта:", silhouette_avg)
joblib.dump(kmeans, "small.joblib")