import numpy as np
import serie
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.distributions import ECDF
from statistics import multimode, median

X = np.array([50, 49, 48, 51, 52, 53, 54 ,57, 59, 60, 61, 55, 60, 62, 63])
Y = np.array([10, 8, 10, 9, 10, 12, 13, 15, 16, 18, 20, 17, 21, 25, 24 ])
n = len(X)

X.sort()
Y.sort()

x_sample_average_x= np.zeros(len(X))
x_sample_average_x2= np.zeros(len(X))
y_y_cherta= np.zeros(len(X))
y_y_cherta2= np.zeros(len(X))
x2 = np.zeros(len(X))
xy = np.zeros(len(X))

x_sred = sum(X) / n
y_sred = sum(Y) / n


for i in range (len(X)):
    x_sample_average_x[i] = X[i] - x_sred
    x_sample_average_x2[i] = math.pow(x_sample_average_x[i], 2)

    y_y_cherta[i] = Y[i] - y_sred
    y_y_cherta2[i] = math.pow(y_y_cherta[i], 2)

    x2[i] = math.pow(X[i], 2)
    xy[i] = X[i] * Y[i]

s_x = math.sqrt(sum(x_sample_average_x2) / (n-1))
s_y = math.sqrt(sum(y_y_cherta2) / (n-1))
xy_sred = sum(xy) / n
r = (xy_sred - x_sred*y_sred) / (s_x * s_y)
sum_y_y_cherta2 = sum(y_y_cherta2)
sum_x_2 = sum(x2)

t_p = (abs(r) * math.sqrt(n-2)) / (math.sqrt(1 - math.pow(r, 2)))
t_T = 1.771

print("   \u2022" + 'среднее производственное средство', x_sred)
print("   \u2022" + 'средняя суточная выроботка', y_sred)


if(t_p > t_T):
    print("   \u2022" + 't_p > t_T, выборочный коэфициент корреляции значимо отлицается от нуля')

t_y = 2.15

sigma_r = (1 - math.pow(r, 2)) / (math.sqrt(n-2))

dover_left = r - t_y * sigma_r
dover_right = r + t_y * sigma_r

print("   \u2022" + 'с вероятностью 0,95 линейный коэффициент корреляции генеральной совокупности находится в пределах от %.2f до %.2f' % (dover_left, dover_right))
d_proc = dover_left * 100
print("   \u2022" + 'Применительно к решаемой задаче полученный результат означает, что по имеющейся выборке следует ожидать влияние производственных средст на рост суточной выработки не менее чем на %.0f процентов' % (d_proc))

regres_y = np.zeros(len(X))
regres_x = np.zeros(len(X))
# for i in range(len(X)):
#     regres_y[i]= y_sred + r * (s_y/s_x) * (X[i] - x_sred)
#     regres_x[i] = x_sred + r * (s_x/s_y) * (Y[i] - y_sred)
for i in range(len(X)):
    regres_y[i] = X[i] * 1.002450 - 40.536231
    regres_x[i] = Y[i] * 0.826076 + 43.043646
    
c_y = x_sred * 1.002450 - 40.536231
c_x = y_sred * 0.826076 + 43.043646

r2 = round(math.pow(r, 2) * 100)
poc_r2 = 100-r2
print("   \u2022" + '%f процентов рассеивания производственных средств объясняется линейной корреляционной зависимостью между средствами и суточной выработке, и только %f рассеивания средств остались необъяснимыми. Такое положение могло произойти из-за того, что в модель не включены другие факторы, влияющие на изменение Х, либо опытных данных в данной выбрке не достаточно, чтобы построить более надежное уравнение регрессии ')

y_galochka = np.zeros(len(X))
y_y_galochka = np.zeros(len(X))
y_y_galochka2 = np.zeros(len(X))
for i in range(n):
    y_galochka[i] = y_sred + r * (s_y/s_x) * (X[i] - x_sred)
    # x_galochka[i] = x_sred + r * (s_x/s_y) * (Y[i] - y_sred)
    y_y_galochka[i] = Y[i] - y_galochka[i]
    y_y_galochka2[i] = math.pow(y_y_galochka[i], 2)

sum_y_y_galochka2 = sum(y_y_galochka2)
R2 = 1 - sum_y_y_galochka2 / sum_y_y_cherta2
F_h = (R2 * (n-2)) / (1-R2)
F_t = 3.81

if(F_h> F_t):
    print("   \u2022" + 'F_H > F_T (%.2f > %.2f)б уравнение линейной регрессии y = x * 1.002 - 40.53 статически значимо описывает результаты эксперимента' % (F_h, F_t))

u = y_y_cherta
u_cherta = sum_y_y_galochka2 / n

u_u_cherta = np.zeros(len(X))
u_u_cherta2 = np.zeros(len(X))

for i in range(n):
    u_u_cherta[i] = u[i] - u_cherta
    u_u_cherta2[i] = math.pow(u_u_cherta[i], 2)

sum_u_u_cherta2 = sum(u_u_cherta2)

sigma_u = math.sqrt(sum_u_u_cherta2 / (n-2))

delta = round(sigma_u / y_sred * 100)
print("   \u2022" + 'так как величина дельта велика (%.0f), прогнозы модели значительно отличаются от фактических значений ' % (delta))


a0 = 43.043646
a1 = 0.826076

S_galochka_y = sum_y_y_galochka2

sum_x = sum(X)

S_y_x = S_galochka_y * math.sqrt(1 - math.pow(r, 2))
S_a_1 = S_y_x * math.sqrt(n / (n*sum_x_2 - math.pow(sum_x, 2)))
S_a_0 = S_y_x * math.sqrt(sum_x_2 / (n*sum_x_2 - math.pow(sum_x, 2)))
if((S_a_0 < abs(a0)) and (S_a_1 < abs(a1))):
    print("   \u2022" + 'коэффициенты а0 и а1 значимы')
   
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

    # ax1.set_xticks(intervals)
    # y, edges, _ = ax1.hist(list, bins=intervals, histtype="bar", edgecolor = 'black', color='#9773ff', label='интер вр')
    
    # ax1.plot(X, regres_x, color='#beff73', marker='o', label='Y на X')
    # ax1.plot(Y, regres_y, color='#9773ff', marker='*', label='X на Y')
    plt.scatter(Y, regres_x, color='#beff73', marker='o', label='Точки')
    plt.plot(Y, regres_line, color='#9773ff', label='Линия тренда')
    plt.scatter(X, regres_y, color='#9773ff', marker='o', label='Точки')
    plt.plot(X, regres_line1, color='#beff73', label='Линия тренда')
    # ax1.plot(x_sred, c_x, color='#9773ff', marker='*', label='сред')
    # ax1.plot(y_sred, c_y, color='#beff73', marker='o', label='сред')

    ax1.legend(loc='best')
    ax1.set_title('линейная регресия')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.grid(color='grey')
    plt.show()
