import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры
num_samples = 100  # Количество выборок
time_interval = np.linspace(0, 1, num_samples)  # Временные точки

# Имитация шестимерного векторного случайного процесса E
E = np.zeros((6, num_samples))

# Генерация случайных величин для каждой компоненты EI
for i in range(6):
    t_i = np.random.uniform(1, time_interval + i, num_samples)
    E[i, :] = t_i

# Визуализация процесса
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
for i in range(6):
    plt.plot(time_interval, E[i, :], label=f'E{i+1}(t)')

plt.title('Шестимерный векторный случайный процесс')
plt.xlabel('Время (t)')
plt.ylabel('Значение компоненты')
plt.legend(loc='best')

# Математические ожидания
mean_values = np.mean(E, axis=1)
plt.subplot(1, 3, 2)
plt.plot(range(1, 7), mean_values, marker='o')
plt.title('Математические ожидания')
plt.xlabel('I')
plt.ylabel('E[I](t)')

# Дисперсии
variance_values = np.var(E, axis=1)
plt.subplot(1, 3, 3)
plt.plot(range(1, 7), variance_values, marker='o', color='orange')
plt.title('Дисперсии')
plt.xlabel('I')
plt.ylabel('Var[I](t)')

plt.tight_layout()
plt.show()