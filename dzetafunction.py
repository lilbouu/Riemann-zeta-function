import re
from flint import *
import matplotlib.pyplot as plt
import numpy as np

# Устанавливаем точность
ctx.dps = 200

# Читаем файл с нулями
with open('zetazeros.txt', 'r') as f:
    content = f.read()

# Разбиваем содержимое файла на блоки по одной или более пустым строкам
blocks = re.split(r'\n\s*\n', content.strip())

# Для каждого блока объединяем все строки в одну строку без пробельных символов
t_values_arb = []
for block in blocks:
    s_clean = "".join(block.split())
    try:
        t_val = arb(s_clean)
        t_values_arb.append(t_val)
    except Exception as e:
        print("Ошибка преобразования строки:", s_clean, ":", e)

# Проверяем, что найдено хотя бы 100 чисел
if len(t_values_arb) < 100:
    raise ValueError(f"Найдено корректных чисел: {len(t_values_arb)}. Проверьте формат входного файла.")

# Берем первые 100 значений (положительных нулей)
t_values_arb = t_values_arb[:100]

# Создаем нули s_i: для каждого положительного нуля добавляем и его отрицательную копию
s_values_acb = [acb(0.5, t) for t in t_values_arb] + [acb(0.5, -t) for t in t_values_arb]
# Общее число строк системы: 100 + 100 = 200

# Задаем число коэффициентов n так, чтобы оно было на 1 больше числа строк
n = 201

# Строим матрицу A размером (200 x 201)
A = acb_mat(200, n)
for i in range(200):
    s_i = s_values_acb[i]
    for k in range(1, n+1):
        A[i, k-1] = (k ** (-s_i))

# Создаем расширенную матрицу A' размером (201 x 201)
A_prime = acb_mat(n, n)
for i in range(200):
    for j in range(n):
        A_prime[i, j] = A[i, j]
# Последняя строка задает дополнительное условие
A_prime[200, 0] = acb(1)
for j in range(1, n):
    A_prime[200, j] = acb(0)

# Создаем вектор B' размером (201 x 1)
B_prime = acb_mat(n, 1)
for i in range(200):
    B_prime[i, 0] = acb(0)
B_prime[200, 0] = acb(1)

# Решаем систему
X = A_prime.solve(B_prime)

# Получаем и выводим коэффициенты (их будет 201)
a_coeffs = [X[i, 0] for i in range(n)]
for k, a in enumerate(a_coeffs, start=1):
    print(f"a_{k} = {a.real} + {a.imag}i")

# Извлекаем действительные, мнимые части и модули коэффициентов
k_values = list(range(1, n+1))
a_real = [float(a.real) for a in a_coeffs]
a_imag = [float(a.imag) for a in a_coeffs]
a_abs = [float(abs(a)) for a in a_coeffs]

# Строим графики для коэффициентов
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(k_values, a_real, 'bo-', label='Re(a_k)')
axs[0].set_xlabel('k')
axs[0].set_ylabel('Re(a_k)')
axs[0].set_title('Действительные части коэффициентов a_k')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(k_values, a_imag, 'ro-', label='Im(a_k)')
axs[1].set_xlabel('k')
axs[1].set_ylabel('Im(a_k)')
axs[1].set_title('Мнимые части коэффициентов a_k')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(k_values, a_abs, 'go-', label='|a_k|')
axs[2].set_xlabel('k')
axs[2].set_ylabel('|a_k|')
axs[2].set_title('Модули коэффициентов a_k')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig("coefficients.png")

# Функция R(s)
def R(s, a_coeffs):
    total = acb(0)
    for k in range(1, len(a_coeffs) + 1):
        total += a_coeffs[k-1] * (k ** (-s))
    return total

# Генерируем точки x от -6 до 0
x_values = [arb(x) for x in np.linspace(-10, 0, 1000)]
s_values = [acb(x, 0) for x in x_values]

R_values = [R(s, a_coeffs) for s in s_values]
R_real = [val.real for val in R_values]

x_floats = [float(x) for x in x_values]
R_real_floats = [float(r) for r in R_real]

# Вычисляем R(s) в тривиальных нулях
trivial_zeros = [0, -2, -4]
R_trivial = [R(acb(x, 0), a_coeffs) for x in trivial_zeros]
for x, R_val in zip(trivial_zeros, R_trivial):
    print(f"R({x}) = {R_val.real} + {R_val.imag}i")

plt.figure()
plt.plot(x_floats, R_real_floats, label='Re(R(s))')
plt.xlabel('x')
plt.ylabel('Re(R(s))')
plt.title('График Re(R(s)) для s = x + 0i, x ∈ [-6, 0]')
plt.grid(True)
plt.legend()
plt.savefig("Direhle.png")
