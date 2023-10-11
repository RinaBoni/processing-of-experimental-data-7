import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import math 



print('\n\n\n')

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
            
    y_cherta_x[i] = sum / n_x[i]
    
nx_x = np.zeros(ar_lengh)
nx_x_2 = np.zeros(ar_lengh)
nxy_xy = np.zeros(ar_lengh)
ny_y = np.zeros(ar_lengh)

for i in range(ar_lengh):
    nx_x[i] = X[i] * n_x[i]
    nx_x_2[i] = math.pow(nx_x[i], 2)
    ny_y[i] = Y[i] * n_y[i]
    for j in range (ar_lengh):
        if(np.isnan(n_ij[j][i])):
            pass
        else:
            nxy_xy[i] = X[i] * Y[j] * n_ij[j][i]
            
sum_nx_x = np.sum(nx_x)
sum_nx_x_2 = np.sum(nx_x_2)
sum_nxy_xy = np.sum(nxy_xy)
sum_ny_y = np.sum(ny_y)

n = 100
a_0 = 63.0768657122
a_1 = 0.02076656019
print('%.0f a_0 + %.0f a_1 = %.0f' % (n, sum_nx_x, sum_ny_y))
print('%.0f a_0 + %.0f a_1 = %.0f' % (sum_nx_x, sum_nx_x_2, sum_nxy_xy))

print('уравнение линейной регрессии:  y_x = %f x + %f' % (a_0, a_1))

y_cherta_x = np.zeros(ar_lengh)
for i in range(ar_lengh):
    y_cherta_x[i] = a_0 * X[i] + a_1

max_element=0
for i in range(ar_lengh):
    for j in range(ar_lengh):
        if(np.isnan(n_ij[j][i])):
            pass
        else:
            if n_ij[j][i] > max_element:
                max_element = n_ij[j][i]
                i_max = i
                j_max = j

MoX = X[i_max]
MoY = Y[j_max]

h1 = 10
h2 = 1

u = np.zeros(ar_lengh)
v = np.zeros(ar_lengh)
u_n_x = np.zeros(ar_lengh)
v_n_y = np.zeros(ar_lengh)
nu_u2 = np.zeros(ar_lengh)
nv_v2 = np.zeros(ar_lengh)
nuv_u_v = np.zeros(ar_lengh)

for i in range (ar_lengh):
    u[i] = (X[i] - MoX) / h1
    v[i] = (Y[i] - MoY) / h2
    u_n_x[i] = n_x[i] * u[i]
    v_n_y[i] = n_y[i] * v[i]
    nu_u2[i] = n_x[i] * math.pow(u[i], 2)
    nv_v2[i] = n_y[i] * math.pow(v[i], 2)
    for j in range (ar_lengh):
        if(np.isnan(n_ij[j][i])):
            pass
        else:
            nuv_u_v[i] =  u[i] * v[j] * n_ij[j][i]
    
sum_nuv_u_v = np.sum(nuv_u_v)    
u_cherta = 1/n * np.sum(u_n_x)
v_cherta = 1/n * np.sum(v_n_y)
u_cherta2 = 1/n * np.sum(nu_u2)
v_cherta2 = 1/n * np.sum(nv_v2)
    
Su = math.sqrt(u_cherta2 - math.pow(u_cherta, 2))    
Sv = math.sqrt(v_cherta2 - math.pow(v_cherta, 2))  

r = (sum_nuv_u_v - n * u_cherta * v_cherta) / (n * Su * Sv)

x_cherta = u_cherta * h1 + MoX
y_cherta = v_cherta * h2 + MoY
Sx = Su * h1
Sy = Sv * h2


y_galochka_x = np.zeros(ar_lengh)
x_galochka_y = np.zeros(ar_lengh)
for i in range(ar_lengh):
    y_galochka_x[i] = 0.00864059 * X[i] + 44.40998137
    x_galochka_y[i] = 0.00109685 * Y[i] + 165.44971996


print('\nиспользуя коэффициент линейной корреляции:')

print('уравнение линии регрессии у на х: y = 0.00864059 * x + 44.40998137')
print('уравнение линии регрессии х на у: x = 0.00109685 * у + 165.44971996\n')


t_H = (abs(r) * math.sqrt(n-2)) / (math.sqrt(1 - math.pow(r, 2)))

t = 1.660

print("   \u2022" + '𝑟 > 0, что говорит о положительной корреляции величин X, Y')
print("   \u2022" + 'r = 0÷0,2, линейной связи нет')

if(t_H>t):
    print("   \u2022" + 't_H > t, выборочный коэффициент линейной корреляции rв значимо отличается от нуля')
if(t_H<t):
    print("   \u2022" + 't_H < t, выборочный коэффициент линейной корреляции rв не значимо отличается от нуля\n     Следовательно, можно считать, что отклонение притока нефти при различных режимах работы (с замерами забойных давлений на грубине манометром) не связаны линейной корреляционной зависимостью.')

y_y_cherta = np.zeros(ar_lengh)
y_y_cherta2 = np.zeros(ar_lengh)
y_cherta_x_y_cherta = np.zeros(ar_lengh)
y_cherta_x_y_cherta2 = np.zeros(ar_lengh)

for i in range(ar_lengh):
    y_y_cherta[i] = Y[i] - y_cherta
    y_y_cherta2[i] = math.pow(y_y_cherta[i], 2)
    y_cherta_x_y_cherta[i] = y_cherta_x[i] - y_cherta
    y_cherta_x_y_cherta2[i] = math.pow(y_cherta_x_y_cherta[i], 2)

Q = np.sum(y_y_cherta2)
Q_r = np.sum(y_cherta_x_y_cherta2)

Qe = Q - Q_r

F_h = (Q_r * (n-2)) / (Qe * (4-1))

F_t = 3.94


print("   \u2022" + 'F_H = %0.3f < %f, модель линейной регрессии y = 0.00864059 * x + 44.40998137 не соглуется с опытными данными')









print('\n\n\n')



# # Выполняем линейную регрессию
slope, intercept = np.polyfit(X, y_galochka_x, 1)

# # Создаем массив значений для линии тренда
regres_line = slope * X + intercept

# Выполняем линейную регрессию
slope1, intercept1= np.polyfit(Y, x_galochka_y, 1)

# Создаем массив значений для линии тренда
regres_line1 = slope1 * Y + intercept1

with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)

    ax1.scatter(X, y_galochka_x, color='#beff73', marker='o')
    ax1.plot(X, regres_line, color='#beff73')
    
    ax1.set_title('уравнение линии регрессии у на х')
    ax1.grid(color='grey')

    ax2.scatter(Y, x_galochka_y, color='#9773ff', marker='o')
    ax2.plot(Y, regres_line1, color='#9773ff')
    
    ax2.set_title('уравнение линии регрессии х на у')
    ax2.grid(color='grey')
    
    ax3.scatter(X, y_galochka_x, color='#beff73')
    ax3.plot(X, regres_line, color='#beff73')
    ax3.scatter(Y, x_galochka_y, color='#9773ff', marker='o' )
    ax3.plot(Y, regres_line1, color='#9773ff')
    
    ax3.set_title('совмещенные')
    ax3.grid(color='grey')
    
    plt.show()


