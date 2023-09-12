import math
from decimal import Decimal, ROUND_HALF_UP
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
                57, 32, 7, 28, 23, 20, 45, 18, 29, 25]

S = sum(list)
S2 = round(math.pow(S, 2))

list.sort();
print(list)
print("Количество всех значений: %.f" % len(list))
print("Сумма всех значений: ", S);
print("Сумма квадратов всех значений: ", S2);
print("Выборочное среднее: ",S/len(list));
print("Выборочное среднее для квадратов: ",S2/len(list));
print("Выборочная дисперсия: ",S2/len(list)-math.pow(S/len(list),2))
print("Выборочная СКО: ",math.sqrt(S2/len(list)-math.pow(S/len(list),2)));
print("Исправленная дисперсия: ",len(list)*(S2/len(list)-math.pow(S/len(list),2))/(len(list)-1));
print("Исправленная СКО: ", math.sqrt(len(list)*(S2/len(list)-math.pow(S/len(list),2))/(len(list)-1)))
 
#part 2
k=math.ceil(1+math.log2(len(list))); # кол-во интервалов
print("Количество интервалов",k);
 
#нахождение и округление наименьшего и наибольшего
x_min=Decimal(list[0]);
x_min=x_min.quantize(Decimal("1.00"), ROUND_HALF_UP);
x_max=Decimal(list[len(list)-1]);
x_max=x_max.quantize(Decimal("1.00"), ROUND_HALF_UP);

R = (x_max - x_min).quantize(Decimal("1.00"), ROUND_HALF_UP)

h =(R/k).quantize(Decimal("1.00"), ROUND_HALF_UP)

print("Начало интервалов x_min: ",x_min)
print("Наибольшее округленное x_max: ",x_max)
print('размах варьирования признака: ', R)
print('длина частичных интервалов: ', h)

#вывод интервалов,середины интервалов,выборочное среднее
xn=x_min
xsred=0
x2sred=0
for i in range(1,k):
    n=0
    xe=xn+((x_max-x_min)/k).quantize(Decimal("1.00"), ROUND_HALF_UP)
    for j in range(int(len(list))):
        if (xn<=list[j]<xe):
            n+=1
    xsred+=((xe+xn)/2)*n #для выборочного среднего
    x2sred+=math.pow((xe+xn)/2,2)*n # для выбороч квадрата среднего
    print("Граница интервала N%.0f: [%.2f - %.2f) принадлежит %.0f чисел, его середина - %.2f"%(i,xn,xe,n,(xe+xn)/2))
    xn=xe
xe=xn+((x_max-x_min)/k).quantize(Decimal("1.00"), ROUND_HALF_UP);
n=0
for j in range(len(list)):
    if (xn <= list[j] <= xe):
        n += 1
print("Граница интервала N%.0f: [%.2f - %.2f) принадлежит %.0f чисел, его середина - %.2f" % (k, xn, xe,n,(xe+xn)/2))
xsred+=((xe+xn)/2)*n #для выборочного среднего
x2sred+=math.pow((xe+xn)/2,2)*n # для выбороч квадрата среднего
print("Выборочное среднее: %f, Выборочное среднее для квадратов: %f"%(xsred/len(list),x2sred/len(list)))
 
#Часть 3.
print("Часть 3. Выборочная дисперсия: ",x2sred/len(list)-math.pow(xsred/len(list),2))
print("Выборочное СКО: ",math.sqrt(x2sred/len(list)-math.pow(xsred/len(list),2)));
print("Исправленная дисперсия: ",len(list)*(x2sred/len(list)-math.pow(xsred/len(list),2))/(len(list)-1))
# print("Исправленная СКО",math.sqrt(len(list)*(x2sred/len(list)-math.po