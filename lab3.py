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

# print('Для Х')
# print(X)

# #начало интервала
# x_start=serie.minimum(X) - 0.5 * serie.h(X)
# # x_start=x_min

# #массив середин интервалов
# X_interval_middle_array = []
# #массив частот интервалов
# X_frequency_array  = []
# #массив с интервалами
# X_intervals = []




# #все, кроме последнего интерава
# for i in range(1,serie.k(X)):
#     X_interval_frequency =0
#     #конец интервала
#     x_end=x_start+(serie.h(X))
#     #считаем сколько элементов попало в интервал
#     for j in range(int(n)):
#         if (x_start<=X[j]<x_end):
#             X_interval_frequency +=1
#     #считаем среднее значение интервала
#     X_interval_middle = (x_end+x_start)/2
    
#     #добавляем значения в массивы
#     X_interval_middle_array.append(X_interval_middle)
#     X_frequency_array .append(X_interval_frequency)
#     X_intervals.append(x_start)
    
#     # print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,x_start,x_end,X_interval_frequency ,X_interval_middle))
#     #новое начало = конец старого
#     x_start=x_end
    

# #последний интервал    
# # x_end=x_max + 0.5 * h;
# x_end=x_start+(serie.h(X));
# X_interval_frequency =0
# for j in range(n):
#     if (x_start <= X[j] <= x_end):
#         X_interval_frequency  += 1
# X_interval_middle = (x_end+x_start)/2
# X_interval_middle_array.append(X_interval_middle)
# X_frequency_array .append(X_interval_frequency)
# X_intervals.append(x_start)
# X_intervals.append(x_end)
# # print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (serie.k(X), x_start, x_end,X_interval_frequency ,X_interval_middle))


# # print('середины интервалов: ', X_interval_middle_array)
# # print('частоты: ', X_frequency_array )
# # print(X_intervals)
# X_interval_middle_array = np.array(X_interval_middle_array)
# X_frequency_array = np.array(X_frequency_array)
# X_intervals = np.array(X_intervals)

# #относительные частоты
# X_relative_frequencies =  X_frequency_array / n
# ar_lengh = serie.ar_lengh(X_frequency_array)
# #накопительные относительные частоты
# X_cumulative_relative_frequencies = np.zeros(ar_lengh)
# for i in range(ar_lengh):
#     if i==0:
#         X_cumulative_relative_frequencies[i] = 0 + X_relative_frequencies[i]
#     else:
#         X_cumulative_relative_frequencies[i] = X_cumulative_relative_frequencies[i-1] + X_relative_frequencies[i]
               
        

# X_max_index = np.argmax(X_frequency_array)
# X_mode_M_o_X = X_interval_middle_array[X_max_index]

# #условные варианты
# X_conditional_options = np.zeros(ar_lengh)
# for i in range(ar_lengh):
#     X_conditional_options[i] = (X_interval_middle_array[i] - X_mode_M_o_X) / serie.h(X)

# #расчетная таблица 10
# X_n_u = np.zeros(ar_lengh)
# X_n_u2 = np.zeros(ar_lengh)
# X_n_u3 = np.zeros(ar_lengh)
# X_n_u4 = np.zeros(ar_lengh)
# X_n_u_1_2 = np.zeros(ar_lengh)
    
# for i in range(ar_lengh):
#     X_n_u[i] = X_frequency_array[i] * X_conditional_options[i]
#     X_n_u2[i] = X_frequency_array[i] * math.pow(X_conditional_options[i], 2)
#     X_n_u3[i] = X_frequency_array[i] * math.pow(X_conditional_options[i], 3)
#     X_n_u4[i] = X_frequency_array[i] * math.pow(X_conditional_options[i], 4)
#     X_n_u_1_2[i] = X_frequency_array[i] *  math.pow((X_conditional_options[i] + 1), 2)


# #контроль вычислений
# X_sum_X_n_u = np.sum(X_n_u)
# X_sum_X_n_u2 = np.sum(X_n_u2)
# X_sum_X_n_u3 = np.sum(X_n_u3)
# X_sum_X_n_u4 = np.sum(X_n_u4)
# X_sum_X_n_u_1_2 = np.sum(X_n_u_1_2)


# if (n + 2 * X_sum_X_n_u + X_sum_X_n_u2) == X_sum_X_n_u_1_2:
#     print('контроль вычислений по таблице 10 пройден')
# # else:
#     # print('lox')
    
# #условные начальные моменты
# X_M1 = X_sum_X_n_u / n
# X_M2 = X_sum_X_n_u2 / n


# #выборочная средняя
# X_sample_average_x = X_M1 * serie.h(X) + X_mode_M_o_X

# #выборочная дисперсия
# X_sample_variance_S2 = (X_M2 - math.pow(X_M1, 2)) * math.pow(serie.h(X), 2)

# #выборочное среднее квадратическое отклонение
# X_sample_mean_square_deviation_S = math.sqrt(X_sample_variance_S2)

# print('выборочная средняя x`: %.2f' % (X_sample_average_x))
# print('выборочное среднее квадратическое отклонение S: %.2f' % (X_sample_mean_square_deviation_S))










# print('Для Y')
# print(Y)

# #начало интервала
# y_start=serie.minimum(Y) - 0.5 * serie.h(Y)
# # y_start=x_min

# #массив середин интервалов
# Y_interval_middle_array = []
# #массив частот интервалов
# Y_frequency_array  = []
# #массив с интервалами
# Y_intervals = []




# #все, кроме последнего интерава
# for i in range(1,serie.k(Y)):
#     Y_interval_frequency =0
#     #конец интервала
#     y_end=y_start+(serie.h(Y))
#     #считаем сколько элементов попало в интервал
#     for j in range(int(n)):
#         if (y_start<=Y[j]<y_end):
#             Y_interval_frequency +=1
#     #считаем среднее значение интервала
#     Y_interval_middle = (y_end+y_start)/2
    
#     #добавляем значения в массивы
#     Y_interval_middle_array.append(Y_interval_middle)
#     Y_frequency_array .append(Y_interval_frequency)
#     Y_intervals.append(y_start)
    
#     # print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,y_start,y_end,Y_interval_frequency ,Y_interval_middle))
#     #новое начало = конец старого
#     y_start=y_end
    

# #последний интервал    
# # y_end=x_max + 0.5 * h;
# y_end=y_start+(serie.h(Y));
# Y_interval_frequency =0
# for j in range(n):
#     if (y_start <= Y[j] <= y_end):
#         Y_interval_frequency  += 1
# Y_interval_middle = (y_end+y_start)/2
# Y_interval_middle_array.append(Y_interval_middle)
# Y_frequency_array .append(Y_interval_frequency)
# Y_intervals.append(y_start)
# Y_intervals.append(y_end)
# # print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (serie.k(Y), y_start, y_end,Y_interval_frequency ,Y_interval_middle))


# # print('середины интервалов: ', Y_interval_middle_array)
# # print('частоты: ', Y_frequency_array )
# # print(Y_intervals)
# Y_interval_middle_array = np.array(Y_interval_middle_array)
# Y_frequency_array = np.array(Y_frequency_array)
# Y_intervals = np.array(Y_intervals)

# #относительные частоты
# Y_relative_frequencies =  Y_frequency_array / n
# ar_lengh = serie.ar_lengh(Y_frequency_array)
# #накопительные относительные частоты
# Y_cumulative_relative_frequencies = np.zeros(ar_lengh)
# for i in range(ar_lengh):
#     if i==0:
#         Y_cumulative_relative_frequencies[i] = 0 + Y_relative_frequencies[i]
#     else:
#         Y_cumulative_relative_frequencies[i] = Y_cumulative_relative_frequencies[i-1] + Y_relative_frequencies[i]
               
        

# Y_max_index = np.argmax(Y_frequency_array)
# Y_mode_M_o_Y = Y_interval_middle_array[Y_max_index]

# #условные варианты
# Y_conditional_options = np.zeros(ar_lengh)
# for i in range(ar_lengh):
#     Y_conditional_options[i] = (Y_interval_middle_array[i] - Y_mode_M_o_Y) / serie.h(Y)

# #расчетная таблица 10
# Y_n_u = np.zeros(ar_lengh)
# Y_n_u2 = np.zeros(ar_lengh)
# Y_n_u3 = np.zeros(ar_lengh)
# Y_n_u4 = np.zeros(ar_lengh)
# Y_n_u_1_2 = np.zeros(ar_lengh)
    
# for i in range(ar_lengh):
#     Y_n_u[i] = Y_frequency_array[i] * Y_conditional_options[i]
#     Y_n_u2[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 2)
#     Y_n_u3[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 3)
#     Y_n_u4[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 4)
#     Y_n_u_1_2[i] = Y_frequency_array[i] *  math.pow((Y_conditional_options[i] + 1), 2)


# #контроль вычислений
# Y_sum_Y_n_u = np.sum(Y_n_u)
# Y_sum_Y_n_u2 = np.sum(Y_n_u2)
# Y_sum_Y_n_u3 = np.sum(Y_n_u3)
# Y_sum_Y_n_u4 = np.sum(Y_n_u4)
# Y_sum_Y_n_u_1_2 = np.sum(Y_n_u_1_2)


# if (n + 2 * Y_sum_Y_n_u + Y_sum_Y_n_u2) == Y_sum_Y_n_u_1_2:
#     print('контроль вычислений по таблице 10 пройден')
# # else:
# #     print('lox')
    
# #условные начальные моменты
# Y_M1 = Y_sum_Y_n_u / n
# Y_M2 = Y_sum_Y_n_u2 / n


# #выборочная средняя
# Y_sample_average_x = Y_M1 * serie.h(Y) + Y_mode_M_o_Y

# #выборочная дисперсия
# Y_sample_variance_S2 = (Y_M2 - math.pow(Y_M1, 2)) * math.pow(serie.h(Y), 2)

# #выборочное среднее квадратическое отклонение
# Y_sample_mean_square_deviation_S = math.sqrt(Y_sample_variance_S2)

# print('выборочная средняя y`: %.2f' % (Y_sample_average_x))
# print('выборочное среднее квадратическое отклонение S: %.2f' % (Y_sample_mean_square_deviation_S))

x_sample_average_x= np.zeros(len(X))
x_sample_average_x2= np.zeros(len(X))
y_sample_average_y= np.zeros(len(X))
y_sample_average_y2= np.zeros(len(X))
x2 = np.zeros(len(X))
xy = np.zeros(len(X))

x_sred = sum(X) / n
y_sred = sum(Y) / n


for i in range (len(X)):
    x_sample_average_x[i] = X[i] - x_sred
    x_sample_average_x2[i] = math.pow(x_sample_average_x[i], 2)

    y_sample_average_y[i] = Y[i] - y_sred
    y_sample_average_y2[i] = math.pow(y_sample_average_y[i], 2)

    x2[i] = math.pow(X[i], 2)
    xy[i] = X[i] * Y[i]

s_x = math.sqrt(sum(x_sample_average_x2) / (n-1))
s_y = math.sqrt(sum(y_sample_average_y2) / (n-1))
xy_sred = sum(xy) / n
r = (xy_sred - x_sred*y_sred) / (s_x * s_y)

t_p = (abs(r) * math.sqrt(n-2)) / (math.sqrt(1 - math.pow(r, 2)))
t_T = 1.771

print('среднее производственное средство', x_sred)
print('средняя суточная выроботка', y_sred)


if(t_p > t_T):
    print('t_p > t_T, выборочный коэфициент корреляции значимо отлицается от нуля')

t_y = 2.15

sigma_r = (1 - math.pow(r, 2)) / (math.sqrt(n-2))

dover_left = r - t_y * sigma_r
dover_right = r + t_y * sigma_r

print('с вероятностью 0,95 линейный коэффициент корреляции генеральной совокупности находится в пределах от %.2f до %.2f' % (dover_left, dover_right))
d_proc = dover_left * 100
print('Применительно к решаемой задаче полученный результат означает, что по имеющейся выборке следует ожидать влияние производственных средст на рост суточной выработки не менее чем на %.0f процентов' % (d_proc))

ragres_y = np.zeros(len(X))
ragres_x = np.zeros(len(X))

for i in range(len(X)):
    ragres_y = y_sred + r * (s_y/s_x) * (X[i] - x_sred)
    ragres_x = x_sred + r * (s_x/s_y) * (Y[i] - y_sred)
    
