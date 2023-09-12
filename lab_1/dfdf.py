import math
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#part 1
list=[]
i=True
S=0
S2=0
# print("Введите числа в столбик, в конце пустую строку")
# while i:
#    i=input();
#    ii=i.replace(",", ".");
#    if i:
#        list.append(float(ii));
#        S+=float(ii);
#        S2+=math.pow(float(ii),2);

list = [52, 33, 10, 22, 28, 34, 39, 29, 21, 27, 
                31, 12, 28, 40, 46, 51, 44, 32, 16, 11, 
                29, 31, 38, 44, 31, 24, 9, 17, 32, 41, 
                47, 31, 42, 15, 21, 29, 50, 55, 37, 19, 
                56, 32, 7, 28, 23, 20, 45, 18, 29, 25]



list.sort();
print(list)

################################################################################################

######################                  пункт 1                  ###############################

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



#начало интервала
x_start=x_min
x_sred=0
x_2sred=0
#массив середин интервалов
interval_middle_array = []
#массив частот
frequency_array = []
#массив с интервалами
intervals = []


#все, кроме последнего интерава
for i in range(1,k):
    frequency =0
    #конец интервала
    x_end=x_start+(h)
    #считаем сколько элементов попало в интервал
    for j in range(int(n)):
        if (x_start<=list[j]<x_end):
            frequency +=1
    x_sred+=((x_end+x_start)/2)*frequency  #для выборочного среднего
    x_2sred+=math.pow((x_end+x_start)/2,2)*frequency  # для выбороч квадрата среднего
    #считаем среднее значение интервала
    interval_middle = (x_end+x_start)/2
    
    #добавляем значения в массивы
    interval_middle_array.append(interval_middle)
    frequency_array.append(frequency)
    intervals.append(x_start)
    
    print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f"%(i,x_start,x_end,frequency ,interval_middle))
    #новое начало = конец старого
    x_start=x_end
    

#последний интервал    
x_end=x_start+(h);
frequency =0
for j in range(n):
    if (x_start <= list[j] <= x_end):
        frequency  += 1
interval_middle = (x_end+x_start)/2
interval_middle_array.append(interval_middle)
frequency_array.append(frequency)
intervals.append(x_start)
intervals.append(x_end)
print("Граница интервала N%.0f: [%.0f - %.0f) принадлежит %.0f чисел, его середина - %.0f" % (k, x_start, x_end,frequency ,interval_middle))


################################################################################################

######################                  пункт 2                  ###############################

#частоты в numpy массив
frequency_array_np = np.array(frequency_array)

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

######################               вывод пункта 1              ###############################

print("Количество всех значений n: %.f" % n)
print("наименьшая варианта выборочной совокупности x_min: ",x_min)
print("наибольшая варианта выборочной совокупности x_max: ",x_max)
print('размах варьирования признака R: ', R)
print("число интервалов вариационного ряда k",k);
print('длина частичных интервалов h: ', h)

print('середины интервалов: ', interval_middle_array)
print('частоты: ', frequency_array)

################################################################################################

######################               вывод пункта 2              ###############################

print('относительные частоты: ', relative_frequencies)
print('накопительные относительные частоты: ', cumulative_relative_frequencies)

################################################################################################

with plt.style.context("dark_background"):


    # fig, axes = plt.subplots(2)
    figure = plt.figure()
    ax1 = figure.add_subplot(2, 1, 1)
    ax2 = figure.add_subplot(2, 1, 2)

    ######################              графики пункта 1             ###############################
    # ax1.style.use('dark_background')

    plt.xticks(intervals)
    y, edges, _ = ax1.hist(list, bins=intervals, histtype="bar", edgecolor = 'black', color='#9773ff', label='интервальный вр')
    ax1.plot(interval_middle_array, frequency_array, color='#beff73', marker='o', label='дискретный вр')
    # ax1.plot(interval_middle_array, frequency_array, color='#beff73', marker='o')

    ax1.legend(loc='best')
    ax1.set_title('гистограмма и полигон', fontsize='20')
    ax1.set_xlabel('середины интервалов')
    ax1.set_ylabel('частоты')
    ax1.grid(color='grey')
    # ax1.show()

    ################################################################################################

    ######################              графики пункта 2             ###############################

    plt.xticks(edges)
    ax2.plot(interval_middle_array, cumulative_relative_frequencies, marker='o' , color='#beff73', label='кумулята')
    ax2.set_title('кумулятивная кривая', fontsize='20')
    ax2.set_xlabel('середины интервалов')
    ax2.set_ylabel('накопительные относительные частоты')
    ax2.legend(loc='best')
    ax2.grid(color='grey')
    plt.show()

################################################################################################

# x_sred+=((x_end+x_start)/2)*frequency  #для выборочного среднего
# x_2sred+=math.pow((x_end+x_start)/2,2)*frequency  # для выбороч квадрата среднего
# print("Выборочное среднее: %.0f, Выборочное среднее для квадратов: %.0f"%(x_sred/n,x_2sred/n))


 
# #Часть 3.
# print("Часть 3. Выборочная дисперсия: ",x_2sred/n-math.pow(x_sred/n,2))
# print("Выборочное СКО: ",math.sqrt(x_2sred/n-math.pow(x_sred/n,2)));
# print("Исправленная дисперсия: ",n*(x_2sred/n-math.pow(x_sred/n,2))/(n-1))







# print("Исправленная СКО",math.sqrt(n*(x_2sred/n-math.po

# def map_to_list(map, result):
#     """проходимся по мапе и заносим данные из нее в список"""
#     #индекс списка
#     cur_row = 0

#     #проходимся по мапе и заносим данные из нее в список
#     for id in map.keys():
#         result.append([])
#         result[cur_row].append(str(map[id]["interval middle"]))
#         result[cur_row].append(str(map[id]["frequency"]))

#         cur_row += 1

#     return result



# def to_dataframe(result, filename):
#     """заносим все в датафрейм и в csv файл"""
#     #заносим все в датафрейм
#     df = pd.DataFrame(result, columns = ["interval middle", "frequency"])
#     #создаем файл csv
    
#     #заносим датафрейм в csv файл
#     df.to_csv(filename)
    
# result = []

# result = map_to_list(map, result)
# to_dataframe(result, 'lab_1/discrete_variation_series.csv')

# df_disc = pd.read_csv('lab_1/discrete_variation_series.csv')

# intervals = [7, 14, 21, 28, 35, 42, 49, 56]


# S = sum(list)
# S2 = round(math.pow(S, 2))

# print("Сумма всех значений: ", S);
# print("Сумма квадратов всех значений: ", S2);
# print("Выборочное среднее: ",S/n);
# print("Выборочное среднее для квадратов: ",S2/n);
# print("Выборочная дисперсия: ",S2/n-math.pow(S/n,2))
# print("Выборочная СКО: ",math.sqrt(S2/n-math.pow(S/n,2)));
# print("Исправленная дисперсия: ",n*(S2/n-math.pow(S/n,2))/(n-1));
# print("Исправленная СКО: ", math.sqrt(n*(S2/n-math.pow(S/n,2))/(n-1)))