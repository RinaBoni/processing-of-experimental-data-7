import numpy as np
import matplotlib.pyplot as plt

# Ваши данные X и Y
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Рассчитываем коэффициенты регрессии Y на X
slope_Y_X, intercept_Y_X = np.polyfit(X, Y, 1)

# Рассчитываем коэффициенты регрессии X на Y
slope_X_Y, intercept_X_Y = np.polyfit(Y, X, 1)

# Создаем массивы для линий тренда
regres_Y_X = slope_Y_X * X + intercept_Y_X
regres_X_Y = slope_X_Y * Y + intercept_X_Y

# Строим графики
fig, ax1 = plt.subplots()
ax1.plot(X, regres_Y_X, color='#beff73', marker='o', label='Y на X')
ax1.plot(Y, regres_X_Y, color='#9773ff', marker='*', label='X на Y')

# Остальной код для настройки графика (названия осей, легенда и т. д.)

# Показываем график
plt.show()