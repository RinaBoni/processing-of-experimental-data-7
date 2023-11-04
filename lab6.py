
from sklearn.cluster import KMeans 

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

# Получите список всех доступных цветов
colors = list(mcolors.TABLEAU_COLORS.keys())

# Выберите первые 10 цветов
ten_colors = colors[:10]


# Генерация 50 случайных величин в интервале (3, 8) для каждой координаты
number_of_realizations = 50
dimension = 9
X = np.random.uniform(low=3, high=8, size=(number_of_realizations, dimension))

# Создание графика scatter


model = KMeans(n_clusters=6, init='k-means++', max_iter=10, random_state=42)
model.fit(X)
lables = model.predict(X)
lables[:10]


with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)

    # ax1.figure(figsize=(8, 8))
    ax1.scatter(X[:, 0], X[:, 1], )

    # Укажите другие координаты, если нужно, например: X[:, 1], X[:, 2], и т. д.

    ax1.set_title('50 случайных реализаций')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.grid(True)
    # ax1.show()

    model.cluster_centers_
    # ax2.figure(figsize=(8, 8))
    for i in range(len(lables)):
        ax2.scatter(X[i, 0], X[i, 1], color=ten_colors[lables[i]])

    for i, centr in enumerate(model.cluster_centers_):
        ax2.scatter(centr[0], centr[1], marker='*', s=500, c=ten_colors[i])
        ax2.text(centr[0]-0.1, centr[1]-0.2, f'{i+1}')
    
    
plt.show()

# centroid_ids = np.random.choice(X.shape[0], k, re)

# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# # plt.plot(range(1,11), wcss)
# # plt.title('Метод локтя')
# # plt.xlabel('Количество кластеров')
# # plt.ylabel('WCSS')
# # plt.show()

# kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)
# y_kmeans = kmeans.fit_predict(X)

# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Кластер 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Кластер 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Кластер 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Кластер 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Кластер 5')
# plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'yellow', label = 'Кластер 6')
# plt.scatter(kmeans.cluster_centers_[0:, 0], kmeans.cluster_centers_[:, 1], s = 300, c='black')
# plt.show()