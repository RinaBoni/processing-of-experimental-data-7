import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.distributions import ECDF
from statistics import multimode, median
# import seaborn as sns


print('\n\n######################################################################################################')

print('######################                  Лабораторная 1                 ###############################\n\n')
print('данные о вводе в эксплуатацию новых газовых скважин за год по различным районам страны: \n')

#part 1
list=[]
i=True
S=0
S2=0


list = [52, 33, 10, 22, 28, 34, 39, 29, 21, 27, 
                31, 12, 28, 40, 46, 51, 44, 32, 16, 11, 
                29, 31, 38, 44, 31, 24, 9, 17, 32, 41, 
                47, 31, 42, 15, 21, 29, 50, 55, 37, 19, 
                57, 32, 7, 28, 23, 20, 45, 18, 29, 25]



list.sort();
print(list)

################################################################################################

######################                  пункт 1                  ###############################


print('\n######################               вывод пункта 1              ###############################\n')


#объем выборки
n = len(list)
#нахождение наименьшего и наибольшего
x_min=list[0];
x_max=list[n-1];
#размах варьирования признака
R = (x_max - x_min)
#число интервалов вариационного ряда
k=round(math.sqrt(n))
#длина частичных интервалов
h =round(R/k)


################################################################################################

######################               вывод пункта 1              ###############################


print("Количество всех значений n: %.f" % n)
print("наименьшая варианта выборочной совокупности x_min: ",x_min)
print("наибольшая варианта выборочной совокупности x_max: ",x_max)
print('размах варьирования признака R: ', R)
print("число интервалов вариационного ряда k",k);
print('длина частичных интервалов h: ', h)


#начало интервала
x_start=x_min - 0.5 * h
#x_start=x_min

#массив середин интервалов
interval_middle_array = []
#массив частот интервалов
frequency_array  = []
#массив с интервалами
intervals = []




#все, кроме последнего интерава
for i in range(1,k):
    interval_frequency =0
    #конец интервала
    x_end=x_start+(h)
    #считаем сколько элементов попало в интервал
    for j in range(int(n)):
        if (x_start<=list[j]<x_end):
            interval_frequency +=1
    #считаем среднее значение интервала
    interval_middle = (x_end+x_start)/2
    
    #добавляем значения в массивы
    interval_middle_array.append(interval_middle)
    frequency_array .append(interval_frequency)
    intervals.append(x_start)
    
    print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,x_start,x_end,interval_frequency ,interval_middle))
    #новое начало = конец старого
    x_start=x_end
    

#последний интервал    
x_end=x_max + 0.5 * h;
#x_end=x_start+(h);
interval_frequency =0
for j in range(n):
    if (x_start <= list[j] <= x_end):
        interval_frequency  += 1
interval_middle = (x_end+x_start)/2
interval_middle_array.append(interval_middle)
frequency_array .append(interval_frequency)
intervals.append(x_start)
intervals.append(x_end)
print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (k, x_start, x_end,interval_frequency ,interval_middle))


print('середины интервалов: ', interval_middle_array)
print('частоты: ', frequency_array )



################################################################################################

######################                  пункт 2                  ###############################

#частоты в numpy массив
frequency_array_np = np.array(frequency_array )

#относительные частоты
relative_frequencies =  frequency_array_np / n

#накопительные относительные частоты
cumulative_relative_frequencies = np.empty(len(frequency_array_np))
for i in range(len(frequency_array_np)):
    if i==0:
        cumulative_relative_frequencies[i] = 0 + relative_frequencies[i]
    else:
        cumulative_relative_frequencies[i] = cumulative_relative_frequencies[i-1] + relative_frequencies[i]
        
################################################################################################

######################               вывод пункта 2              ###############################

print('\n\n\n######################               вывод пункта 2              ###############################\n')
print('относительные частоты: ', relative_frequencies)
print('накопительные относительные частоты: ', cumulative_relative_frequencies)
        
        
        
        
        
################################################################################################

######################                  пункт 3                  ###############################

list_np = np.array(list)
#используем существующую эмпирицескую функцию распределения из библиотеки statsmodels
ecdf = ECDF(list_np)





################################################################################################

######################                  пункт 4                  ###############################

print('\n\n\n######################               вывод пункта 4              ###############################\n')

#мода
interval_middle_array_np = np.array(interval_middle_array)
# mode_M_o_X = multimode(interval_middle_array_np)
#медиана
median_M_e_X = median(interval_middle_array_np)

#частоты(кажого элемента)
# values, frequency_array = np.unique(list_np, return_counts=True)
# print ('частоты: ', frequency_array)


max_index = np.argmax(frequency_array_np)
mode_M_o_X = interval_middle_array_np[max_index]

#условные варианты
conditional_options = np.empty(len(interval_middle_array_np))
for i in range(len(conditional_options)):
    conditional_options[i] = (interval_middle_array_np[i] - mode_M_o_X) / h

#расчетная таблица 10
n_u = np.empty(len(conditional_options))
n_u2 = np.empty(len(conditional_options))
n_u3 = np.empty(len(conditional_options))
n_u4 = np.empty(len(conditional_options))
n_u_1_2 = np.empty(len(conditional_options))
    
for i in range(len(conditional_options)):
    n_u[i] = frequency_array_np[i] * conditional_options[i]
    n_u2[i] = frequency_array_np[i] * math.pow(conditional_options[i], 2)
    n_u3[i] = frequency_array_np[i] * math.pow(conditional_options[i], 3)
    n_u4[i] = frequency_array_np[i] * math.pow(conditional_options[i], 4)
    n_u_1_2[i] = frequency_array_np[i] *  math.pow((conditional_options[i] + 1), 2)


#контроль вычислений
sum_n_u = np.sum(n_u)
sum_n_u2 = np.sum(n_u2)
sum_n_u3 = np.sum(n_u3)
sum_n_u4 = np.sum(n_u4)
sum_n_u_1_2 = np.sum(n_u_1_2)


if (n + 2 * sum_n_u + sum_n_u2) == sum_n_u_1_2:
    print('контроль вычислений по таблице 10 пройден')
else:
    print('lox')
    
#условные начальные моменты
M1 = sum_n_u / n
M2 = sum_n_u2 / n
M3 = sum_n_u3 / n
M4 = sum_n_u4 / n


#выборочная средняя
sample_average_x = M1 * h + mode_M_o_X

#выборочная дисперсия
sample_variance_S2 = (M2 - math.pow(M1, 2)) * math.pow(h, 2)

#выборочное среднее квадратическое отклонение
sample_mean_square_deviation_S = math.sqrt(sample_variance_S2)

#коэффициент вариации
coefficient_variation_V = sample_mean_square_deviation_S / sample_average_x

#центральные моменты третьего и четвертого порядков
m3 = (M3 - 3*M2*M1 + 2*M1) * math.pow(h, 3)
m4 = (M4 - 4*M3*M1 + 6*M2*math.pow(M1, 2) - 3*math.pow(M1, 4)) * math.pow(h, 4)

#асимметрия
asymmetry_A_S = m3 / math.pow(sample_mean_square_deviation_S, 3)
#эксцесс
excess_E_x = m4 / math.pow(sample_mean_square_deviation_S, 4)

################################################################################################

######################               вывод пункта 4              ###############################


print('мода MoX: ', mode_M_o_X)
print('медиана MeX: ', median_M_e_X)
print('условные начальные моменты: M*1 = %.2f, M*2 = %.2f, M*3 = %.2f, M*4 = %.2f' % (M1, M2, M3, M4))
print('выборочная средняя x`: ', sample_average_x)
print('выборочная дисперсия S^2: %.2f' % (sample_variance_S2))
print('выборочное среднее квадратическое отклонение S: %.2f' % (sample_mean_square_deviation_S))
print('коэффициент вариации V: %.2f' % (coefficient_variation_V))
print('условный центральный момент третьего порядка: ', m3)
print('условный центральный момент четвертого порядка: ', m4)
print('асимметрию As: %.2f' % (asymmetry_A_S))
print('эксцесс Ex: %.2f' % (excess_E_x))


################################################################################################

######################                  пункт 5                  ###############################

t_gamma = 1.984

general_average_confidence_interval_left = sample_average_x - sample_mean_square_deviation_S/math.sqrt(n)*t_gamma
general_average_confidence_interval_rigth = sample_average_x + sample_mean_square_deviation_S/math.sqrt(n)*t_gamma
 
q = 0.143
 
general_standard_deviation_confidence_interval_left = sample_mean_square_deviation_S * (1 - q)
general_standard_deviation_confidence_interval_rigth = sample_mean_square_deviation_S * (1 + q)

################################################################################################

######################               вывод пункта 5              ###############################
print('\n\n\n######################               вывод пункта 5              ###############################\n')
print('уровень надежности гамма: 0.95')
print('t_gamma: ', t_gamma)
print('средняя обводненность нефти должна находиться в промежутке (%.2f;%.2f)' % (general_average_confidence_interval_left, general_average_confidence_interval_rigth))
print('q = %.3f, q < 1' % (q) )
print('отклонения истинных значений обводненности нефти не должны выходить за пределы промежутка (%.2f %.2f)' % (general_standard_deviation_confidence_interval_left, general_standard_deviation_confidence_interval_rigth))




print('\n\n\n################################################################################################')

print('######################                  Лабораторная 2                  ###############################\n')

xi__x = np.empty(len(frequency_array_np))
ui = np.empty(len(frequency_array_np))
f_ui = np.empty(len(frequency_array_np))
yi = np.empty(len(frequency_array_np))
ni = np.empty(len(frequency_array_np))

for i in range(len(frequency_array_np)):
    xi__x[i] = interval_middle_array_np[i] - sample_average_x
    ui[i] = xi__x[i] / sample_mean_square_deviation_S
    f_ui[i] = ui[i]
    yi[i] = n*h / sample_mean_square_deviation_S * f_ui[i]
    ni[i] = round(yi[i])

print(yi)
print(ni)

################################################################################################

######################                   графики                 ###############################

with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(2, 2, 1)
    ax2 = figure.add_subplot(2, 2, 2)
    ax3 = figure.add_subplot(2, 2, 3)

################################################################################################

######################              графики пункта 1             ###############################
 

    ax1.set_xticks(intervals)
    y, edges, _ = ax1.hist(list, bins=intervals, histtype="bar", edgecolor = 'black', color='#9773ff', label='интервальный вр')
    ax1.plot(interval_middle_array, frequency_array , color='#beff73', marker='o', label='дискретный вр')
    ax1.legend(loc='best')
    ax1.set_title('гистограмма и полигон', fontsize='16')
    ax1.set_xlabel('середины интервалов', fontsize=12)
    ax1.set_ylabel('частоты', fontsize=12)
    ax1.grid(color='grey')

################################################################################################

######################              графики пункта 2             ###############################

    ax2.set_xticks(intervals)
    ax2.plot(interval_middle_array, cumulative_relative_frequencies, marker='o' , color='#beff73', label='кумулята')
    ax2.set_title('кумулятивная кривая', fontsize='16')
    ax2.set_xlabel('середины интервалов', fontsize=12)
    ax2.set_ylabel('накопительные относительные частоты', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(color='grey')


################################################################################################

######################              графики пункта 3             ###############################

    ax3.set_xticks(intervals)
    ax3.step(ecdf.x, ecdf.y, color='#beff73', label='эфр')
    ax3.set_title('эмпирическая функция распределения', fontsize='16')
    ax3.grid(color='grey')
    ax3.set_ylabel('$F(x)$', fontsize=12)
    ax3.set_xlabel('$x$', fontsize=12)
    ax3.legend(loc='best')


    ################################################################################################
    # plt.show()
################################################################################################