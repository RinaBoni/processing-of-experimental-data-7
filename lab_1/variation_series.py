from collections import Counter
import numpy as np
import math
# import tabulate
import pandas as pd 
import scipy as sp
from scipy import stats  # модуль статистических функций
import matplotlib.pyplot as plt

#################################################################################################

######################                  создание                  ###############################

def create_array_data():
        '''создаем массив с данными и сортируем его по возвастанию'''
        #массив с данными
        data = np.array([52, 33, 10, 22, 28, 34, 39, 29, 21, 27, 
                31, 12, 28, 40, 46, 51, 44, 32, 16, 11, 
                29, 31, 38, 44, 31, 24, 9, 17, 32, 41, 
                47, 31, 42, 15, 21, 29, 50, 55, 37, 19, 
                57, 32, 7, 28, 23, 20, 45, 18, 29, 25])
        #сортируем массив по возрастанию
        data = np.sort(data, axis=None)
        return data

def save_csv(data):
        ''''''
        #заголовок для csv
        header_for_data = ['water content of oil']
        #создаем датафрейм для данных
        df = pd.DataFrame(data)
        #сохраняем массив в csv
        df.to_csv("lab_1/data.csv", header=header_for_data, index=False)
        
def c_s_csv():
        save_csv(create_array_data())
        
################################################################################################

######################                  работа с csv                  ###############################


def load_data():
    '''Загрузить данные'''
    return pd.read_csv("lab_1/data.csv")



def data_columns():
    '''Получить имена полей кадра данных'''
    return load_data().columns



def colum_values():
    '''Получить значения поля "обводненность нефти"'''
    return load_data()['water content of oil']



def colum_values_unique():
    '''Получить значения в поле "обводненность нефти" без дубликатов'''
    return load_data()['water content of oil'].unique()



def numuber_of_unique():
    '''Рассчитать частоты в поле "обводненность нефти" 
       (количества появлений разных значений)'''
    return Counter( load_data()['water content of oil'] )



################################################################################################

######################                  пункт 1                  ###############################
def x_max(xs):
    '''наибольшая варианта выборочной совокупности'''
    return xs.max()

def r_x_max():
        '''Вернуть наибольшую варианту выборочной совокупности поля "обводненность нефти"'''
        return x_max( load_data()['water content of oil'] )



def x_min(xs):
    '''наименьшая варианта выборочной совокупности'''
    return xs.min()

def r_x_min():
        '''Вернуть наименьшую варианту выборочной совокупности поля "обводненность нефти"'''
        return x_min( load_data()['water content of oil'] )



def R(xs):
        '''размах варьирования признака'''
        return x_max(xs) - x_min(xs)

def r_R():
        '''Вернуть размах варьирования признака поля "обводненность нефти"'''
        return R( load_data()['water content of oil'] )



def n(xs):
        '''объем выборки'''
        return xs.size

def r_n():
        '''Вернуть объем выборки поля "обводненность нефти"'''
        return n( load_data()['water content of oil'] )



def k(xs):
        '''число интервалов вариационного ряда'''
        return round(math.sqrt(n(xs)))

def r_k():
        '''Вернуть число интервалов вариационного ряда поля "обводненность нефти"'''
        return k( load_data()['water content of oil'] )



def h(xs):
        '''длина частичных интервалов'''
        return round(R(xs)/k(xs))

def r_h():
        '''Вернуть длину частичных интервалов поля "обводненность нефти"'''
        return h( load_data()['water content of oil'] )



def nbin(n, xs): 
    '''Разбивка данных на частотные корзины'''
    min_x, max_x = min(xs), max(xs)
    range_x = max_x - min_x
    fn = lambda x: min( int((abs(x) - min_x) / range_x * n), n-1 )
    return map(fn, xs)

def ex_1_11():
        '''Разбиmь электорат Великобритании на 5 корзин'''
        series = load_data()['water content of oil']
        return Counter( nbin(5, series) )
        # return  list(nbin(5, series) )

print('по корзинам: ', ex_1_11())
################################################################################################

######################                  пункт 4                  ###############################

def mean(xs): 
    '''Среднее значение числового ряда'''
    return sum(xs) / len(xs) 

def r_mean():
    '''Вернуть среднее значение поля "обводненность нефти"'''
    return mean( load_data()['water content of oil'] )



def median(xs):
    '''Медиана числового ряда'''
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return sorted(xs)[mid]
    else:
        return mean( sorted(xs)[mid-1:][:2] )

def r_median():
    '''Вернуть медиану поля "обводненность нефти"'''
    return median( load_data()['water content of oil'] )



def variance(xs):
    '''Дисперсия (варианс) числового ряда,
       несмещенная дисперсия при n <= 30'''
    mu = mean(xs)
    n = len(xs)
    n = n-1 if n in range(1, 30) else n  
    square_deviation = lambda x : (x - mu) ** 2 
    return sum( map(square_deviation, xs) ) / n

def r_variance():
    '''Вернуть дисперсию поля "обводненность нефти"'''
    return variance( load_data()['water content of oil'] )



def standard_deviation(xs):
    '''Стандартное отклонение числового ряда'''
    return np.sqrt( variance(xs) )
       
def r_standard_deviation():
    '''Вернуть стандартное отклонение поля "обводненность нефти"'''
    return standard_deviation( load_data()['water content of oil'] )



##############################################################################################

######################                  вывод                  ###############################

print('Получить значения поля "обводненность нефти"\n', colum_values())
print('Рассчитать частоты в поле "обводненность нефти" (количества появлений разных значений)', numuber_of_unique())

print('наибольшая варианта выборочной совокупности: ', r_x_max())
print('наименьшая варианта выборочной совокупности: ', r_x_min())
print('размах варьирования признака: ', r_R())
print('объем выборки: ', r_n())
print('число интервалов вариационного ряда: ', r_k())
print('длина частичных интервалов: ', r_h())


print('среднее значение ', r_mean() )
print('медиана', r_median())
print('дисперсия: ', r_variance())
print('квадратичное отклонение: ', r_standard_deviation())





# n_i = {}
# for i in data:
#         if i in n_i:
#                 n_i[i] += 1
#         else:
#                 n_i[i] = 1

# for i, j in n_i.items():
#         print(f"{i}: {j}")

# print('из интернета:')

# from itertools import groupby

# arr = [1.112, 1.113, 1.114, 1.111, 1.221, 1.223, 1.321, 1.021, 1.03, 2.0, 3.6, 4.2]
# print(arr)


# result = groupby(arr, key=lambda x: int(x * 10) if 1 <= x < 2 else -1)

# for key, vals in result:
#     print(key / 10, list(vals))
    
    
