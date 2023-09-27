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
                56, 32, 7, 28, 23, 20, 45, 18, 29, 25]



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
ar_lengh = len(frequency_array_np)

#относительные частоты
relative_frequencies =  frequency_array_np / n

#накопительные относительные частоты
cumulative_relative_frequencies = np.zeros(ar_lengh)
for i in range(ar_lengh):
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
conditional_options = np.zeros(ar_lengh)
for i in range(ar_lengh):
    conditional_options[i] = (interval_middle_array_np[i] - mode_M_o_X) / h

#расчетная таблица 10
n_u = np.zeros(ar_lengh)
n_u2 = np.zeros(ar_lengh)
n_u3 = np.zeros(ar_lengh)
n_u4 = np.zeros(ar_lengh)
n_u_1_2 = np.zeros(ar_lengh)
    
for i in range(ar_lengh):
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

xi__x = np.zeros(ar_lengh)
ui = np.zeros(ar_lengh)
f_ui = np.zeros(ar_lengh)
yi = np.zeros(ar_lengh)
ni = np.zeros(ar_lengh)

#таблица 16
for i in range(ar_lengh):
    xi__x[i] = round((interval_middle_array_np[i] - sample_average_x), 2)
    ui[i] = round((xi__x[i] / sample_mean_square_deviation_S), 2)
    f_ui[i] = round((1 / (math.sqrt(2 * math.pi)) * math.exp(-(math.pow(ui[i], 2) / 2))), 4)
    yi[i] = round((n*h / sample_mean_square_deviation_S * f_ui[i]), 1)
    ni[i] = int(round(yi[i]))
    
#################################################################

##########           критерий Пирсона         ###################

n__ni = np.zeros(ar_lengh)
n__ni2 = np.zeros(ar_lengh)
n__ni2__ni = np.zeros(ar_lengh)
#хи в квадрате
hi_square = 0

number_of_degrees_of_freedom_k = k - 3

for i in range(ar_lengh):
    n__ni[i] = frequency_array_np[i] - ni[i]
    n__ni2[i] = pow(n__ni[i], 2)
    n__ni2__ni[i] = n__ni2[i] / ni[i]
    hi_square += n__ni2__ni[i]
    
critical_value_hi_square = 0.711

print('критерий Пирсона:')

if (critical_value_hi_square > hi_square):
    print('\n\nX2_0 < X0_кр, нет достаточных оснований отвергнуть выдвинутую гипотезу о нормальном распределении признака Х\n')
else:
    print('\n\nX2_0 < X0_кр, гипотеза о нормальном распределении признака Х отвергается\n')

######################################################################

##########           критерий Кольмогорова         ###################

statistics_lambda = round((abs(np.max(frequency_array_np) - np.max(ni)) / math.sqrt(n)), 1)
sum_kolmogor_em = 0

for i in range (1000):
    sum_kolmogor_em = math.pow((-1), number_of_degrees_of_freedom_k) * math.exp(-2 * math.pow(number_of_degrees_of_freedom_k, 2) * math.pow(statistics_lambda, 2))
funk_kolmogor_em = 1 - sum_kolmogor_em

funk_kolmogor_ter = 1.0000

print('критерий Колмогорова:')

if (abs(funk_kolmogor_em - funk_kolmogor_ter)>0.05):
    print('\nразница между эмпирическим распределением и теоретическим равна ', round(abs(funk_kolmogor_em - funk_kolmogor_ter), 3), 'существенное расхождение между эмпирическим и теоретическим распределениями, которое нельзя считать случайным. Следовательно, рассматриваемая выборка не может быть смоделирована нормальным законом распределения\n\n')
else:
    print('\nразница между эмпирическим распределением и теоретическим равна ', round(abs(funk_kolmogor_em - funk_kolmogor_ter), 3), 'расхождение между частотами может быть случайным, и распределения хорошо соответствуют одно другому\n\n')


D_n_plus_ar = np.zeros(ar_lengh)
D_n_minus_ar = np.zeros(ar_lengh)

for i in range(ar_lengh):
    D_n_plus_ar[i] = i/n - 1 + math.exp(-1 * (interval_middle_array_np[i] / sample_average_x))
    D_n_minus_ar[i] = 1 - math.exp(-1 * (interval_middle_array_np[i] / sample_average_x) - (i-1)/n)
    
D_n_plus = np.max(D_n_plus_ar)
D_n_minus = np.max(D_n_minus_ar)
Dn = max(D_n_minus, D_n_plus)

lambda_kolmagor = 1.09

if (((Dn - 0.2/n)*(math.sqrt(n) + 0.26 + 0.5/n)) <= lambda_kolmagor):
    print('неравенство при выбранном λ𝛼 выполняется, эмпирическое распределение можно изучать на математической модели, подчиняющейся экспоненциальному закону распределения')
else:
    print('неравенство при выбранном λ𝛼 не выполняется, эмпирическое распределение нельзя изучать на математической модели, подчиняющейся экспоненциальному закону распределения')

######################################################################

##########           приближенный критерий         ###################

Sasymmetry_A_S = math.sqrt((6 * (n-1))/((n+1)*(n+3)))
Sexcess_E_x = math.sqrt((24 * n*(n-2)*(n-3))/(math.pow((n-1), 2) * (n+3) * (n+5)))


if(abs(asymmetry_A_S) <= Sasymmetry_A_S) and (abs(excess_E_x) <= Sexcess_E_x):
    print(' As ≤ SAs и Ex ≤ SEx, то выборочная совокупность подчиняется нормальному закону распределения')
if(abs(asymmetry_A_S) > Sasymmetry_A_S) and (abs(excess_E_x) > Sexcess_E_x):
    print('As > SAs и Ex > SEx, выборочная совокупность не будет распределена по нормальному закону')
if(abs(asymmetry_A_S) > Sasymmetry_A_S) and (abs(excess_E_x) < Sexcess_E_x):
    print('As > SAs и Ex < SEx, выборочная совокупность не будет распределена по нормальному закону')
if(abs(asymmetry_A_S) < Sasymmetry_A_S) and (abs(excess_E_x) > Sexcess_E_x):
    print('As < SAs и Ex > SEx, выборочная совокупность не будет распределена по нормальному закону')
    
hi_square_pribrej = math.pow(asymmetry_A_S, 2) / math.pow(Sasymmetry_A_S, 2) + math.pow(excess_E_x, 2) / math.pow(Sexcess_E_x, 2)
################################################################################################

######################                   графики                 ###############################

with plt.style.context("dark_background"):


    figure = plt.figure()
    ax1 = figure.add_subplot(2, 3, 1)
    ax2 = figure.add_subplot(2, 3, 2)
    ax3 = figure.add_subplot(2, 3, 3)
    ax4 = figure.add_subplot(2, 3, 4)
    ax5 = figure.add_subplot(2, 3, 5)
    ax6 = figure.add_subplot(2, 3, 6)

################################################################################################

######################              графики пункта 1             ###############################
 

    ax1.set_xticks(intervals)
    y, edges, _ = ax1.hist(list, bins=intervals, histtype="bar", edgecolor = 'black', color='#9773ff', label='интервальный вр')
    ax1.plot(interval_middle_array, frequency_array , color='#beff73', marker='o', label='дискретный вр')
    ax1.legend(loc='best')
    ax1.set_title('гистограмма и полигон')
    ax1.set_xlabel('середины интервалов')
    ax1.set_ylabel('частоты')
    ax1.grid(color='grey')

################################################################################################

######################              графики пункта 2             ###############################

    ax2.set_xticks(intervals)
    ax2.plot(interval_middle_array, cumulative_relative_frequencies, marker='o' , color='#beff73', label='кумулята')
    ax2.set_title('кумулятивная кривая')
    ax2.set_xlabel('середины интервалов')
    ax2.set_ylabel('нак относ частоты')
    ax2.legend(loc='best')
    ax2.grid(color='grey')


################################################################################################

######################              графики пункта 3             ###############################

    ax3.set_xticks(intervals)
    ax3.step(ecdf.x, ecdf.y, color='#beff73', label='эфр')
    ax3.set_title('эмпирическая функция распределения')
    ax3.grid(color='grey')
    ax3.set_ylabel('$F(x)$')
    ax3.set_xlabel('$x$')
    ax3.legend(loc='best')


################################################################################################

######################              графики пункта              ###############################

    ax4.set_xticks(interval_middle_array)
    ax4.plot(interval_middle_array, ni, marker='o' , color='#beff73', label='теоретическая')
    ax4.plot(interval_middle_array, frequency_array , color='#9773ff', marker='o', label='эмпирическая')
    ax4.set_title('эмпирическая кривая распределения')
    ax4.set_xlabel('середины интервалов')
    ax4.set_ylabel('на')
    ax4.legend(loc='best')
    ax4.grid(color='grey')

    ################################################################################################
    # plt.show()
################################################################################################