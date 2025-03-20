from flint import *
import matplotlib.pyplot as plt
import numpy as np

# Устанавливаем точность
ctx.dps = 200

# Читаем первые 25 значений t из файла
with open('zetazeros.txt', 'r') as f:
    lines = f.readlines()
t_values = [float(line.split()[1]) for line in lines[:25]]

# Преобразуем в тип arb и создаем нули s_i
t_values_arb = [arb(t) for t in t_values]
s_values_acb = [acb(0.5, t) for t in t_values_arb] + [acb(0.5, -t) for t in t_values_arb]

# Задаем n
n = 51  # Увеличиваем n до 51, чтобы соответствовать 50 нулям

# Строим матрицу A (50 x 51)
A = acb_mat(50, 51)
for i in range(50):
    s_i = s_values_acb[i]
    for k in range(1, n+1):
        A[i, k-1] = (k ** (-s_i))

# Создаем расширенную матрицу A' (51 x 51)
A_prime = acb_mat(51, 51)
for i in range(50):
    for j in range(51):
        A_prime[i, j] = A[i, j]
A_prime[50, 0] = acb(1)
for j in range(1, 51):
    A_prime[50, j] = acb(0)

# Создаем вектор B' (51 x 1)
B_prime = acb_mat(51, 1)
for i in range(50):
    B_prime[i, 0] = acb(0)
B_prime[50, 0] = acb(1)

# Решаем систему
X = A_prime.solve(B_prime)

# Получаем и выводим коэффициенты
a_coeffs = [X[i, 0] for i in range(51)]
for k, a in enumerate(a_coeffs, start=1):
    print(f"a_{k} = {a.real} + {a.imag}i")

# Извлекаем действительные, мнимые части и модули коэффициентов
k_values = list(range(1, 52))
a_real = [float(a.real) for a in a_coeffs]
a_imag = [float(a.imag) for a in a_coeffs]
a_abs = [float(abs(a)) for a in a_coeffs]

# Строим графики для коэффициентов
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# График для Re(a_k)
axs[0].plot(k_values, a_real, 'bo-', label='Re(a_k)')
axs[0].set_xlabel('k')
axs[0].set_ylabel('Re(a_k)')
axs[0].set_title('Действительные части коэффициентов a_k')
axs[0].grid(True)
axs[0].legend()

# График для Im(a_k)
axs[1].plot(k_values, a_imag, 'ro-', label='Im(a_k)')
axs[1].set_xlabel('k')
axs[1].set_ylabel('Im(a_k)')
axs[1].set_title('Мнимые части коэффициентов a_k')
axs[1].grid(True)
axs[1].legend()

# График для |a_k|
axs[2].plot(k_values, a_abs, 'go-', label='|a_k|')
axs[2].set_xlabel('k')
axs[2].set_ylabel('|a_k|')
axs[2].set_title('Модули коэффициентов a_k')
axs[2].grid(True)
axs[2].legend()

# Сохраняем графики коэффициентов
plt.tight_layout()
plt.savefig("coefficients.png")

# Функция R(s)
def R(s, a_coeffs):
    total = acb(0)  # Инициализируем комплексное число
    for k in range(1, len(a_coeffs) + 1):
        total += a_coeffs[k-1] * (k ** (-s))
    return total

# Генерируем точки x от -6 до 0
x_values = [arb(x) for x in np.linspace(-6, 0, 100)]
s_values = [acb(x, 0) for x in x_values]  # s = x + 0i

# Вычисляем R(s) для каждого s
R_values = [R(s, a_coeffs) for s in s_values]

# Извлекаем вещественную часть
R_real = [val.real for val in R_values]

# Преобразуем в float для построения графика
x_floats = [float(x) for x in x_values]
R_real_floats = [float(r) for r in R_real]

# Вычисляем R(s) в тривиальных нулях
trivial_zeros = [0, -1, -2]
R_trivial = [R(acb(x, 0), a_coeffs) for x in trivial_zeros]

# Выводим значения R(s) в тривиальных нулях
for x, R_val in zip(trivial_zeros, R_trivial):
    print(f"R({x}) = {R_val.real} + {R_val.imag}i")


# Строим график для Re(R(s))
plt.figure()
plt.plot(x_floats, R_real_floats, label='Re(R(s))')
plt.xlabel('x')
plt.ylabel('Re(R(s))')
plt.title('График Re(R(s)) для s = x + 0i, x ∈ [-6, 0]')
plt.grid(True)
plt.legend()
plt.savefig("Direhle.png")