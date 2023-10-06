import numpy as np
import math
def ar_lengh(array):
    """длинна массива"""
    return len(array)
def minimum(array):
    """минимальное значение в массиве"""
    return np.min(array)

def maximum(array):
    """максимальное значение в массиве"""
    return np.max(array)

def R(array):
    """размах варьирования признака"""
    return (maximum(array) - minimum(array))
def k(array):
    """число интервалов вариационного ряда"""
    return round(math.sqrt(ar_lengh(array)))
def h(array):
    """длина частичных интервалов"""
    return round(R(array)/k(array))

def intevals_calc(array):
    """расчет интервалов"""
    interval_middle_array = np.empty(k(array))
    frequency_array = np.empty(k(array))
    intervals = np.empty(k(array))
    x_start = minimum(array) - 0.5 * h(array)
    ka = k(array)
    #все, кроме последнего интерава
    for i in range(1, ka):
        interval_frequency =0
        #конец интервала
        print(i)
        x_end=x_start+(h(array))
        #считаем сколько элементов попало в интервал
        for j in range(ar_lengh(array)):
            if (x_start<=array[j]<x_end):
                interval_frequency +=1
        #считаем среднее значение интервала
        interval_middle = (x_end+x_start)/2
        print(interval_middle)
        #добавляем значения в массивы
        interval_middle_array[i] = interval_middle
        print(interval_middle_array[i])
        frequency_array[i] = interval_frequency
        intervals[i] = x_start
        
        print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,x_start,x_end,interval_frequency ,interval_middle))
        #новое начало = конец старого
        x_start = x_end
        

    #последний интервал    
    x_end = maximum(array) + 0.5 * h(array);
    # x_end = maxmum(array);
    interval_frequency =0
    for j in range(ar_lengh(array)):
        if (x_start <= array[j] <= x_end):
            interval_frequency  += 1
    interval_middle = (x_end+x_start)/2
    print(interval_middle)
    interval_middle_array[3] = interval_middle
    print(interval_middle_array[3])
    frequency_array[3] = interval_frequency
    intervals[2] = x_start
    intervals[3] = x_end
    print(interval_middle_array)
    print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (ka, x_start, x_end,interval_frequency ,interval_middle))
    
    return interval_middle_array, frequency_array, intervals

def rel_freq(frequency_array, n):
    return frequency_array / n
