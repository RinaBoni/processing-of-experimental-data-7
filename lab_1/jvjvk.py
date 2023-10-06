print('Для Х')
print(Y)

#начало интервала
y_start=serie.minimum(Y) - 0.5 * serie.h(Y)
# y_start=x_min

#массив середин интервалов
Y_interval_middle_array = []
#массив частот интервалов
Y_frequency_array  = []
#массив с интервалами
Y_intervals = []




#все, кроме последнего интерава
for i in range(1,serie.k(Y)):
    Y_interval_frequency =0
    #конец интервала
    y_end=y_start+(serie.h(Y))
    #считаем сколько элементов попало в интервал
    for j in range(int(n)):
        if (y_start<=Y[j]<y_end):
            Y_interval_frequency +=1
    #считаем среднее значение интервала
    Y_interval_middle = (y_end+y_start)/2
    
    #добавляем значения в массивы
    Y_interval_middle_array.append(Y_interval_middle)
    Y_frequency_array .append(Y_interval_frequency)
    Y_intervals.append(y_start)
    
    print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,y_start,y_end,Y_interval_frequency ,Y_interval_middle))
    #новое начало = конец старого
    y_start=y_end
    

#последний интервал    
# y_end=x_max + 0.5 * h;
y_end=y_start+(serie.h(Y));
Y_interval_frequency =0
for j in range(n):
    if (y_start <= Y[j] <= y_end):
        Y_interval_frequency  += 1
Y_interval_middle = (y_end+y_start)/2
Y_interval_middle_array.append(Y_interval_middle)
Y_frequency_array .append(Y_interval_frequency)
Y_intervals.append(y_start)
Y_intervals.append(y_end)
print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (serie.k(Y), y_start, y_end,Y_interval_frequency ,Y_interval_middle))


print('середины интервалов: ', Y_interval_middle_array)
print('частоты: ', Y_frequency_array )
print(Y_intervals)
Y_interval_middle_array = np.array(Y_interval_middle_array)
Y_frequency_array = np.array(Y_frequency_array)
Y_intervals = np.array(Y_intervals)

#относительные частоты
Y_relative_frequencies =  Y_frequency_array / n
ar_lengh = serie.ar_lengh(Y_frequency_array)
#накопительные относительные частоты
Y_cumulative_relative_frequencies = np.zeros(ar_lengh)
for i in range(ar_lengh):
    if i==0:
        Y_cumulative_relative_frequencies[i] = 0 + Y_relative_frequencies[i]
    else:
        Y_cumulative_relative_frequencies[i] = Y_cumulative_relative_frequencies[i-1] + Y_relative_frequencies[i]
               
        

Y_max_index = np.argmax(Y_frequency_array)
Y_mode_M_o_Y = Y_interval_middle_array[Y_max_index]

#условные варианты
Y_conditional_options = np.zeros(ar_lengh)
for i in range(ar_lengh):
    Y_conditional_options[i] = (Y_interval_middle_array[i] - Y_mode_M_o_Y) / serie.h(Y)

#расчетная таблица 10
Y_n_u = np.zeros(ar_lengh)
Y_n_u2 = np.zeros(ar_lengh)
Y_n_u3 = np.zeros(ar_lengh)
Y_n_u4 = np.zeros(ar_lengh)
Y_n_u_1_2 = np.zeros(ar_lengh)
    
for i in range(ar_lengh):
    Y_n_u[i] = Y_frequency_array[i] * Y_conditional_options[i]
    Y_n_u2[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 2)
    Y_n_u3[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 3)
    Y_n_u4[i] = Y_frequency_array[i] * math.pow(Y_conditional_options[i], 4)
    Y_n_u_1_2[i] = Y_frequency_array[i] *  math.pow((Y_conditional_options[i] + 1), 2)


#контроль вычислений
Y_sum_Y_n_u = np.sum(Y_n_u)
Y_sum_Y_n_u2 = np.sum(Y_n_u2)
Y_sum_Y_n_u3 = np.sum(Y_n_u3)
Y_sum_Y_n_u4 = np.sum(Y_n_u4)
Y_sum_Y_n_u_1_2 = np.sum(Y_n_u_1_2)


if (n + 2 * Y_sum_Y_n_u + Y_sum_Y_n_u2) == Y_sum_Y_n_u_1_2:
    print('контроль вычислений по таблице 10 пройден')
# else:
    # print('lox')
    
#условные начальные моменты
Y_M1 = Y_sum_Y_n_u / n
Y_M2 = Y_sum_Y_n_u2 / n


#выборочная средняя
Y_sample_average_x = Y_M1 * serie.h(Y) + Y_mode_M_o_Y

#выборочная дисперсия
Y_sample_variance_S2 = (Y_M2 - math.pow(Y_M1, 2)) * math.pow(serie.h(Y), 2)

#выборочное среднее квадратическое отклонение
Y_sample_mean_square_deviation_S = math.sqrt(Y_sample_variance_S2)

print('выборочная средняя x`: %.2f' % (Y_sample_average_x))
print('выборочное среднее квадратическое отклонение S: %.2f' % (Y_sample_mean_square_deviation_S))
