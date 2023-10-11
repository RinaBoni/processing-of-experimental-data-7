import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import math 

X = np.array([125, 135, 145, 155, 165, 175, 185, 195, 205])
Y = np.array([11, 12, 13, 14, 15, 166, 17, 18, 19, ])
n_y = np.array([7, 9, 8, 11, 20, 18, 9, 10, 100])
n_x = np.array([3, 9, 12, 13, 22, 21, 6, 8, 6, 100])
n_ij = np.array([[3, 4, nan, nan, nan, nan, nan, nan, nan,],
                [nan, 5, 4,  nan, nan, nan, nan, nan, nan,],
                [nan, nan, 3, 5, nan, nan, nan, nan, nan,],
                [nan, nan, 5, 6,  nan, nan, nan, nan, nan,],
                [nan, nan, nan, 2, 18, nan, nan, nan, nan,],
                [nan, nan, nan, nan, 4, 14, nan, nan, nan,],
                [nan, nan, nan, nan, nan, 7, 2, nan, nan,],
                [nan, nan, nan, nan, nan, nan, 4, 6, nan,],
                [nan, nan, nan, nan, nan, nan, nan, 2, 6]])

ar_lengh = len(X)

y_cherta_x = np.zeros(ar_lengh)

for i in range(ar_lengh):
    sum = 0
    for j in range(ar_lengh):
        if(np.isnan(n_ij[j][i])):
            pass
        else:
            
            sum = sum + n_ij[j][i] * Y[j]
            
    print('sum: ', sum)
    y_cherta_x[i] = sum / n_x[i]
    
nx_x = np.zeros(ar_lengh)
nx_x_2 = np.zeros(ar_lengh)
nxy_xy = np.zeros(ar_lengh)
ny_y = np.zeros(ar_lengh)

for i in range(ar_lengh):
    nx_x[i] = X[i] * n_x[i]
    nx_x_2[i] = math.pow(nx_x[i], 2)
    ny_y[i] = Y[i] * n_y[i]
    nxy_xy[i] = X[i] * Y[i] * 100

# # Выполняем линейную регрессию
slope, intercept = np.polyfit(X, y_cherta_x, 1)

# # Создаем массив значений для линии тренда
regres_line = slope * X + intercept

# # Выполняем линейную регрессию
# slope1, intercept1= np.polyfit(Y, regres_y, 1)

# # Создаем массив значений для линии тренда
# regres_line1 = slope1 * Y + intercept1

with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(1, 1, 1)

    ax1.scatter(X, y_cherta_x, color='#beff73', label='')
    ax1.plot(X, regres_line, color='#9773ff', label='')

#     plt.scatter(Y, regres_x, color='#beff73', marker='o', label='Точки')
#     plt.plot(Y, regres_line, color='#9773ff', label='Линия тренда')
#     plt.scatter(X, regres_y, color='#9773ff', marker='o', label='Точки')
#     plt.plot(X, regres_line1, color='#beff73', label='Линия тренда')

    ax1.legend(loc='best')
    ax1.set_title('')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.grid(color='grey')
    # plt.show()


