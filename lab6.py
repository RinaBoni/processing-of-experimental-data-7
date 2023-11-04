import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Cluster2D:

    def __init__(self, maxDeviation: float = 0, centroid: set = None, points: list = []):
        self.Centroid = centroid
        self.PreviousCentroid = centroid
        self.MaxDeviation = maxDeviation
        self.Points = list(points)

    def AddPoint(self, point: set):
        if (self.Centroid is None):
            self.Centroid = point
            return
        md = self.MaxDeviation
        if (np.abs(point[0] - self.Centroid[0]) > md 
            or np.abs(point[1] - self.Centroid[1]) > md):
            return
        self.Points.append(point)

    def SetPoints(self, points: list):
        self.Points = points

    def UpdatePoints(self):
        md = self.MaxDeviation
        i = 0 
        while i != len(self.Points):
            p = self.Points[i]
            if (np.abs(p[0] - self.Centroid[0]) > md 
                or np.abs(p[1] - self.Centroid[1]) > md):
                self.Points.remove(p)
            else:
                i += 1

    def UpdateCentroid(self):
        self.Centroid = (np.average([p[0] for p in self.Points]), 
                            np.average([p[1] for p in self.Points]))
        
    def K_Means(self, max_iterations: int = 100):
        self.UpdatePoints()
        if (len(self.Points) < 1):
            return
        for _ in range(max_iterations):
            self.PreviousCentroid = self.Centroid
            self.UpdateCentroid()
            if (self.Centroid == self.PreviousCentroid):
                break
            


LEN_POINTS_GROUP = 9
LEN_DATA = 50
LEN_POINTS = LEN_DATA * LEN_POINTS_GROUP
POINT_COORDINATE_MIN_VALUE = 3
POINT_COORDINATE_MAX_VALUE = 8
CLUSTER_MAX_RADIUS = 0.15

P = np.array([[set() for j in range(LEN_POINTS_GROUP)]
                for i in range(LEN_DATA)])
for i in range(LEN_DATA):
    for j in range(LEN_POINTS_GROUP):
        x = np.random.uniform(POINT_COORDINATE_MIN_VALUE,
                                POINT_COORDINATE_MAX_VALUE)
        y = np.random.uniform(POINT_COORDINATE_MIN_VALUE,
                                POINT_COORDINATE_MAX_VALUE)
        P[i, j] = (x, y)

P_FULL = P.reshape(LEN_POINTS)

with plt.style.context("dark_background"):
    
    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)
    

    for p in P_FULL:
        ax1.scatter(p[0], p[1], c='gray', alpha=0.85)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Points raw view')
    


    P_FULL_LIST = P_FULL.tolist()


    def CalculateMaxes(points: np.ndarray, claster_max_radius: float = CLUSTER_MAX_RADIUS) -> set:
        # Инициализация пустого списка для хранения информации о точках и установка максимального счетчика 'm' в 0
        points_info = []
        m = 0
        
        # Итерация по каждой точке в массиве 'points'
        for p in points:
            ps = []  # Инициализация пустого списка 'ps' для хранения близлежащих точек
            
            # Итерация по всем точкам снова для вычисления расстояний
            for p1 in points:
                distance = np.sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2)  # Вычисление евклидова расстояния
                
                # Если расстояние больше 0 и меньше или равно 'claster_max_radius'
                if (distance > 0 and distance <= claster_max_radius):
                    ps.append(p1)  # Добавление близкой точки в 'ps'
                    
            lps = len(ps)  # Получение количества близлежащих точек
            
            if (lps > 0):
                points_info.append((p, lps))  # Если есть близлежащие точки, добавить точку и количество в 'points_info'
                
            if (lps > m):
                m = lps  # Обновление 'm', если найдено новое максимальное значение
        
        if (len(points_info) > 1):
            # Если есть более одной точки с близлежащими соседями, найти точку с максимальным количеством соседей и вернуть ее
            for pi in points_info:
                if (pi[1] == m):
                    return pi[0]
        return points[0]  # Если у точек нет соседей, вернуть первую точку во входном массиве


    # Инициализация счетчиков и установка начального радиуса
    clusters_counter = 0
    oneclusters_counter = 0
    radius = CLUSTER_MAX_RADIUS

    # Цикл до тех пор, пока все точки не будут обработаны
    while (len(P_FULL_LIST) > 0):
        mp = CalculateMaxes(P_FULL_LIST, radius)  # Найти точку с максимальным количеством соседей в текущем радиусе
        c = Cluster2D(radius, mp, P_FULL_LIST)  # Создать объект Cluster2D с использованием радиуса и выбранной точки
        c.K_Means()  # Применить алгоритм кластеризации K-Means к кластеру
        
        x, y = [], []  # Инициализация списков для хранения координат X и Y кластера
        
        # Сбор точек, принадлежащих текущему кластеру, и удаление их из основного списка
        for p in c.Points:
            P_FULL_LIST.remove(p)
            x.append(p[0])
            y.append(p[1])
            
        lp = len(c.Points)  # Получение количества точек в кластере
        
        # Установка параметров визуализации в зависимости от количества точек в кластере
        info = [25, 0.35, None, 20] if lp > 1 else [10, 1.0, 'red', 10]
        
        # Получение цвета маркера центроида для однородной окраски
        centroid_color = ax2.scatter(
            c.Centroid[0], c.Centroid[1], info[0], info[2], marker='*', alpha=info[1]).get_edgecolor()
        
        # Рассеивание точек кластера с цветом центроида
        ax2.scatter(x, y, info[3], c=centroid_color, alpha=1)
        
        clusters_counter += 1  # Инкремент счетчика кластеров
        
        if (lp == 1):
            oneclusters_counter += 1  # Инкремент счетчика кластеров с одной точкой
            
        clear_output()  # Очистка вывода для лучшей визуализации прогресса
        print(len(P_FULL_LIST))  # Вывод оставшегося количества точек


    # Добавление заголовка и меток к конечному графику
    ax2.set_title(f'Clusters: {clusters_counter}; One-clusters: {oneclusters_counter}')
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    
    
plt.show()