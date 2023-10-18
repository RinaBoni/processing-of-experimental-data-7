import numpy as np
import matplotlib.pyplot as plt
import math

print('\n\n\n')

x = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
y = np.array([12.5, 17.8, 37, 41.9, 45, 47, 39, 32, 28])

n = len(x)

x_sred = sum(x) / n
y_sred = sum(y) / n

x_sample_average_x= np.zeros(len(x))
x_sample_average_x2= np.zeros(len(x))
y_y_cherta= np.zeros(len(x))
y_y_cherta2= np.zeros(len(x))
xy = np.zeros(len(x))

for i in range (len(x)):
    x_sample_average_x[i] = x[i] - x_sred
    x_sample_average_x2[i] = math.pow(x_sample_average_x[i], 2)

    y_y_cherta[i] = y[i] - y_sred
    y_y_cherta2[i] = math.pow(y_y_cherta[i], 2)

    xy[i] = x[i] * y[i]

s_x = math.sqrt(sum(x_sample_average_x2) / (n-1))
s_y = math.sqrt(sum(y_y_cherta2) / (n-1))

xy_sred = sum(xy) / n
r = (xy_sred - x_sred*y_sred) / (s_x * s_y)

regres_y = np.zeros(len(x))
regres_x = np.zeros(len(x))


for i in range(len(x)):
    regres_y[i] = x[i] * 0.84222222 - 25.77555555
    regres_x[i] = y[i] * 0.1761722 + 3.1236782

print('линия тренда напоминает параболу, поэтому используем степенную регрессию')

left = round((math.sqrt(x[0] * x[n-1])), 2)

interpolir = y[0] + (y[1] - y[0]) / (x[1] - x[0]) *(left - x[0])


right = math.sqrt(x[0] * x[n-1])

otklonenie_delta = abs(right - interpolir)

x2 = np.zeros(len(x))
x3 = np.zeros(len(x))
xy = np.zeros(len(x))
x4 = np.zeros(len(x))
x2_y = np.zeros(len(x))

for i in range(n):
    x2[i] = math.pow(x[i], 2)
    x3[i] = math.pow(x[i], 3)
    x4[i] = math.pow(x[i], 4)
    xy[i] = x[i] * y[i]
    x2_y[i] = x2[i] * y[i]
    
sum_y = sum(y)
sum_x = sum(x)
sum_x2 = sum(x2)
sum_x3 = sum(x3)
sum_x4 = sum(x4)
sum_x2_y = sum(x2_y)
sum_xy = sum(xy)

A = np.array([[n, sum_x, sum_x2], [sum_x, sum_x2, sum_x3], [sum_x2, sum_x3, sum_x4]])
B = np.array([sum_y, sum_xy, sum_x2_y])

result = np.linalg.solve(A, B)

a0 = result[0]
a1 = result[1]
a2 = result[2]

print(f'\nуравнение регрессии: y = {a0} + {a1} * x  {a2} * x^2')

y_galochka_x = np.zeros(len(x))
y_y_galochka_x_2 = np.zeros(len(x))
y_y_sred_2 = np.zeros(len(x))
for i in range(n):
    y_galochka_x[i] = a0 + a1 * x[i] + a2 * x2[i]
    y_y_galochka_x_2[i] = math.pow((y[i] - y_galochka_x[i]), 2)
    y_y_sred_2[i] = math.pow((y[i] - y_sred), 2)
    
sum_y_y_galochka_x_2 = sum(y_y_galochka_x_2)
sum_y_y_sred_2 = sum(y_y_sred_2)

S_galochka2_yx = 1 / (n-1) * sum_y_y_galochka_x_2
S_galochka2_y = 1 / (n-1) * sum_y_y_sred_2

I = round(math.sqrt(1 - S_galochka2_yx/S_galochka2_y), 2)

F_h = (math.pow(I, 2) * (n-2)) / (1 - math.pow(I, 2))

F_t = 4.74

print(f'\nF_H > F_t модель адекватна. Следовательно, зависимость мощности на долоте от осевой статической назрузки на забой при бурении пород по данным выборки описывается уравнением y = {a0:.5f} + {a1:.5f} * x  {a2:.5f} * x^2')
print('\n\n\n')
############################# графики
# Выполняем линейную регрессию
slope, intercept = np.polyfit(y, regres_x, 1)

# Создаем массив значений для линии тренда
regres_line = slope * y + intercept

# Выполняем линейную регрессию
slope1, intercept1= np.polyfit(y, regres_y, 1)

# Создаем массив значений для линии тренда
regres_line1 = slope1 * y + intercept1


with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    # ax2 = figure.add_subplot(1, 2, 2)


    ax1.scatter(x, regres_y, color='#9773ff', marker='o', label='Точки')
    ax1.plot(x, regres_line1, color='#9773ff', label='Линия тренда')

    ax1.legend(loc='best')
    ax1.set_title('линейная регресия')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.grid(color='grey')
    
    
    # ax2.plot(x, y_galochka_x, color='#9773ff', label='Линия тренда')
    # ax2.legend(loc='best')
    # ax2.set_title('линейная регресия')
    # ax2.set_xlabel('')
    # ax2.set_ylabel('')
    # ax2.grid(color='grey')
    
    plt.show()