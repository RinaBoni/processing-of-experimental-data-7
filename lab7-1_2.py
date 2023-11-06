import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Заданные точки данных
x = np.array([2.31, 2.65, 3.12, 3.67, 3.99, 4.34, 4.89, 5.34, 5.87, 6.88, 7.13, 7.45, 7.89, 8.41, 8.95])
y = np.array([7.34, 5.54, 6.24, 3.76, 4.98, 3.67, 8.35, 7.55, 4.45, 3.54, 5.34, 5.76, 7.43, 5.65, 6.34])
z = np.array([3.98, 4.54, 3.76, 3.45, 4.34, 5.34, 3.45, 2.34, 4.54, 4.34, 3.76, 4.34, 3.34, 5.76, 4.56])

# Создание сетки для визуализации
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

# Интерполяция для создания поверхности
z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

# Вычисление остатков (ошибок аппроксимации)
residuals = z - griddata((x, y), z, (x, y), method='cubic')

# Построение поверхности
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')

# Построение заданных точек
ax.scatter(x, y, z, c='red', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Добавление оценки ошибки на график
error_estimation = np.mean(np.abs(residuals))
ax.text(4, 5, 2, f'Оценка ошибки: {error_estimation}', color='red', fontsize=12)

plt.show()