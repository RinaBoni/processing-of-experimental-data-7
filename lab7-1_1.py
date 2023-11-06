import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.array([0.35, 0.70, 1.05, 1.40, 1.75, 2.10, 2.45, 2.80, 3.15])
y = np.array([4.15, 4.30, 4.45, 4.60, 4.75, 4.90, 5.05])
z = np.array([[3.87, 4.17, 8.65, 3.19, 4.65, 4.98, 7.12, 6.65, 3.76],
              [4.75, 3.76, 2.19, 5.34, 4.65, 2.14, 4.54, 3.33, 6.54],
              [5.43, 4.24, 5.43, 4.33, 5.33, 3.54, 5.34, 4.32, 3.43],
              [6.33, 3.33, 2.43, 4.54, 5.34, 5.34, 4.54, 5.43, 4.43],
              [5.43, 4.43, 4.54, 3.43, 5.32, 5.34, 6.54, 4.54, 5.54],
              [4.54, 3.54, 4.76, 3.76, 5.65, 4.54, 5.76, 7.54, 3.54],
              [5.76, 7.54, 5.76, 3.23, 4.34, 3.54, 3.23, 5.76, 3.43]])

# Создаем интерполяцию для двумерной сетки
f = interpolate.interp2d(x, y, z, kind='cubic')

# Точка, в которой нужно найти значение функции
x_new, y_new = 1.01, 4.87

# Получаем значение z в точке (1.01, 4.87)
z_interpolated = f(x_new, y_new)

print(f"\n\n\n\nЗначение z в точке (1.01, 4.87): {z_interpolated[0]}\n")

# Оценка погрешности
# Заданные ограничения производных
limit_dx = 5
limit_dy = 6

# Погрешность оценивается через производные и шаги x, y
error = (0.5 * limit_dx * (x[1] - x[0])**2) + (0.5 * limit_dy * (y[1] - y[0])**2)
print(f"Погрешность: {error}\n\n\n\n")

# Построение графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, z, cmap='viridis')

ax.scatter(x_new, y_new, z_interpolated, color='red', s=100, label='(1.01, 4.87)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.show()

