import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры процесса
num_samples = 100  # Количество выборок
time_points = np.linspace(0, 1, num_samples)  # Временные точки

# Создаем массив для хранения компонент процесса
process = np.zeros((6, num_samples))

# Генерация шестимерного векторного случайного процесса
for i in range(6):
    process[i, :] = np.random.uniform(1, time_points + i, num_samples)

# Вычисляем математические ожидания и дисперсии
means = np.mean(process, axis=1)
variances = np.var(process, axis=1)

# Визуализация процесса
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
for i in range(6):
    plt.plot(time_points, process[i, :], label=f'E{i+1}(t)')

plt.title('Шестимерный векторный случайный процесс')
plt.xlabel('Время (t)')
plt.ylabel('Значение компоненты')
plt.legend(loc='best')

# Визуализация математических ожиданий

plt.subplot(1, 3, 2)
plt.plot(range(1, 7), means, marker='o', linestyle='-', color='b')
plt.title('Математические ожидания компонент')
plt.xlabel('Компонента (I)')
plt.ylabel('Математическое ожидание')

# Визуализация дисперсий
plt.subplot(1, 3, 3)
plt.plot(range(1, 7), variances, marker='o', linestyle='-', color='r')
plt.title('Дисперсии компонент')
plt.xlabel('Компонента (I)')
plt.ylabel('Дисперсия')

plt.tight_layout()
plt.show()