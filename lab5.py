import numpy as np
import matplotlib.pyplot as plt
import math


X = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
Y = np.array([12.5, 17.8, 37, 41.9, 45, 47, 39, 32, 28])

n = len(X)

x_sred = sum(X) / n
y_sred = sum(Y) / n

x_sample_average_x= np.zeros(len(X))
x_sample_average_x2= np.zeros(len(X))
y_y_cherta= np.zeros(len(X))
y_y_cherta2= np.zeros(len(X))
xy = np.zeros(len(X))

for i in range (len(X)):
    x_sample_average_x[i] = X[i] - x_sred
    x_sample_average_x2[i] = math.pow(x_sample_average_x[i], 2)

    y_y_cherta[i] = Y[i] - y_sred
    y_y_cherta2[i] = math.pow(y_y_cherta[i], 2)

    xy[i] = X[i] * Y[i]

s_x = math.sqrt(sum(x_sample_average_x2) / (n-1))
s_y = math.sqrt(sum(y_y_cherta2) / (n-1))

xy_sred = sum(xy) / n
r = (xy_sred - x_sred*y_sred) / (s_x * s_y)

regres_y = np.zeros(len(X))
regres_x = np.zeros(len(X))

print(y_sred)
print(x_sred)

print(s_y)
print(s_x)
print(r)

# y
# =
# 0.84222222
# x
# +
# 25.77555555
# for i in range(len(X)):
#     regres_y[i]= y_sred + r * (s_y/s_x) * (X[i] - x_sred)
#     regres_x[i] = x_sred + r * (s_x/s_y) * (Y[i] - y_sred)
for i in range(len(X)):
    regres_y[i] = X[i] * 0.84222222 - 25.77555555
    regres_x[i] = Y[i] * 0.1761722 + 3.1236782

left = round((math.sqrt(X[0] * X[n-1])), 2)
print('lef:', left)

############################# графики
# Выполняем линейную регрессию
slope, intercept = np.polyfit(Y, regres_x, 1)

# Создаем массив значений для линии тренда
regres_line = slope * Y + intercept

# Выполняем линейную регрессию
slope1, intercept1= np.polyfit(Y, regres_y, 1)

# Создаем массив значений для линии тренда
regres_line1 = slope1 * Y + intercept1


with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(1, 1, 1)


    ax1.scatter(X, regres_y, color='#9773ff', marker='o', label='Точки')
    ax1.plot(X, regres_line1, color='#9773ff', label='Линия тренда')

    ax1.legend(loc='best')
    ax1.set_title('линейная регресия')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.grid(color='grey')
    # plt.show()