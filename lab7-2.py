import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Заданные данные
x = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
f = np.array([3.76, 2.67, 2.84, 1.17, 2.39, 4.98, 5.28, 5.91, 4.27, 3.44])

# Создание кубического сплайна
cs = CubicSpline(x, f, bc_type='not-a-knot')

# Значение сплайна в точке x=0.3
x_new = 0.3
y_new = cs(x_new)

# Построение графика
x_plot = np.linspace(0, 2.7, 100)
y_plot = cs(x_plot)

plt.figure(figsize=(8, 6))
plt.plot(x, f, 'o', label='Исходные данные')
plt.plot(x_plot, y_plot, label='Квадратичный сплайн')
plt.plot(x_new, y_new, 'ro', label=f'Spline({x_new}) = {y_new:.2f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Квадратичный сплайн')
plt.grid(True)
plt.show()