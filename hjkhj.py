import numpy as np
from scikit.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, shapiro

# Задаем данные
x = np.array([15, 20, 25, 30, 35])  # Температура (x)
y = np.array([30, 40, 50, 60, 70])       # Средняя скорость (y)

datax = np.array([15,20,20,25,25,25,30,30,30,35,35,35,40])
datay = np.array([30,30,40,40,50,50,50,60,60,60,70,70,70])

# Преобразуем x в формат, подходящий для модели (вектор-столбец)
x = x.reshape(-1, 1)

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель на данных
model.fit(x, y)

# Получаем коэффициенты регрессии
slope = model.coef_[0]
intercept = model.intercept_

# Выводим уравнение регрессии (метод наименьших квадратов)
print(f"Уравнение регрессии (метод наименьших квадратов): y = {slope:.2f}x + {intercept:.2f}")

# Задаем исходные данные в исходной системе координат
x_original = np.array([15, 20, 25, 30, 35])
y_original = np.array([30, 40, 50, 60, 70])

# Предсказываем значения для x
predicted_y = model.predict(x)

# Рассчитываем коэффициент корреляции r
r, p_value = pearsonr(x.flatten(), y)

# Выводим коэффициент корреляции и его значимость
print(f"Выборочный коэффициент корреляции (r): {r:.2f}")
print(f"Значимость коэффициента корреляции (p-value): {p_value:.4f}")

# Оцениваем тесноту связи на основе коэффициента корреляции r и его значимости
if abs(r) >= 0.7 and p_value < 0.05:
    print("Существует сильная и статистически значимая связь между X и Y.")
elif abs(r) >= 0.5 and p_value < 0.05:
    print("Существует умеренная и статистически значимая связь между X и Y.")
else:
    print("Связь между X и Y слабая или статистически незначимая.")

# Выводим уравнение линейной регрессии на основе коэффициента корреляции r
slope_r = r * (np.std(y) / np.std(x))
intercept_r = np.mean(y) - slope_r * np.mean(x)

print(f"Уравнение линейной регрессии (коэффициент корреляции r): y = {slope_r:.2f}x + {intercept_r:.2f}")

# Сравниваем уравнения
if abs(r - slope) < 0.01:
    print("Уравнения линейной регрессии очень близки.")
elif abs(r) > abs(slope):
    print("Уравнение линейной регрессии (коэффициент корреляции r) лучше подходит для данных.")
else:
    print("Уравнение линейной регрессии (метод наименьших квадратов) лучше подходит для данных.")

# Проверяем адекватность модели, записанной через коэффициент корреляции r
y_mean = np.mean(y)
y_pred_r = slope_r * x + intercept_r
ssr_r = np.sum((y_pred_r - y_mean) ** 2)
sst_r = np.sum((y - y_mean) ** 2)
r_squared_r = ssr_r / sst_r

print(f"R-squared для модели (коэффициент корреляции r): {r_squared_r:.2f}")

# Оцениваем адекватность модели на основе R-squared
if r_squared_r >= 0.5:
    print("Модель, записанная через коэффициент корреляции r, является адекватной.")
else:
    print("Модель, записанная через коэффициент корреляции r, не является адекватной.")

# Проверяем надежность уравнения регрессии (коэффициент корреляции r)
residuals_r = y - y_pred_r
stat, p_value_res = shapiro(residuals_r)
alpha = 0.05

print(f"p-value для теста на нормальность остатков: {p_value_res:.4f}")

if p_value_res < alpha:
    print("Остатки не имеют нормальное распределение, уравнение регрессии (коэффициент корреляции r) может быть ненадежным.")
else:
    print("Остатки имеют нормальное распределение, уравнение регрессии (коэффициент корреляции r) надежно.")

# Построение уравнения регрессии в первоначальной системе координат
mean_x_original = np.mean(x_original)
mean_y_original = np.mean(y_original)
std_x_original = np.std(x_original)
std_y_original = np.std(y_original)

slope_original = (std_y_original / std_x_original) * slope_r
intercept_original = mean_y_original - slope_original * mean_x_original

print(f"Уравнение линейной регрессии в первоначальной системе координат: y = {slope_original:.2f}x + {intercept_original:.2f}")

# Строим график данных и регрессии (метод наименьших квадратов)
# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.scatter(x, y, label="Данные")
# plt.plot(x, predicted_y, color='red', label="Линейная регрессия (МНК)")
# plt.xlabel("Температура (x)")
# plt.ylabel("Средняя скорость (y)")
# plt.legend()
# plt.grid(True)

# Строим уравнение регрессии в первоначальной системе координат
#plt.subplot(122)
plt.scatter(datax, datay, label="Данные (исходная)")
plt.plot(x, predicted_y, color='red', label="Линейная регрессия ")
plt.plot(x_original, slope_original * x_original + intercept_original, color='purple', label="Линейная регрессия (исходная)")
plt.xlabel("Температура (x)")
plt.ylabel("Средняя скорость (y)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
