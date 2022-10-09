from scipy import signal

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("hello")

pi = np.pi


def x(t):
    return np.cos(520 * pi * t + pi / 3)


def y(t):
    return np.cos(280 * pi * t - pi / 3)


def z(t):
    return np.cos(120 * pi * t + pi / 3)


t = np.linspace(0, 0.03, 6)

fix, axs = plt.subplots(3)
fix.suptitle("exercitiul 1")

axs[0].plot(t, x(t), color="red")
axs[1].plot(t, y(t), color="blue")
axs[2].plot(t, z(t), color="green")

plt.show()

# 2 a
t = np.linspace(0, 4, 1600)


def a(t):
    return np.sin(4 * t)


plt.title("2 a")
plt.plot(t, a(t), color='red')
plt.show()

# 2 b
t = np.linspace(0, 3, 2400)


def b(t):
    return np.sin(4 * t)


plt.title("2 b")
plt.plot(t, b(t), color='red')
plt.show()

# 2 c
t = np.linspace(0, 1, 240)
plt.title("2 c")
plt.plot(t, signal.sawtooth(2 * pi * 5 * t))
plt.show()
q
# 2 d
t = np.linspace(0, 1, 300)
plt.title("2 d")
plt.plot(t, signal.square(2 * pi * 5 * t))
plt.show()

# 3 e
Z = np.random.randint(128, size=(128, 128))
plt.title("3 e")
plt.imshow(Z)

plt.show()

# 3 f
F = np.random.randint(128, size=(128, 128))

for i in range(0, 128):
    for j in range(0, 128):
        if (F[i][j] < 64):
            F[i][j] = 0
        else:
            F[i][j] = 128

plt.title("3 f")
plt.imshow(F)
plt.show()

# 3 a
# Daca frecventa de esentionare este de 2000 Hz, atunci intervalul de timp intre doua esantioane este de:
# 1s = 2000HZ
# 1000ms = 20000HZ
# distanta intre 2 esantioane este de: 1000 / 2000 = 0.5 ms = 0.0005 s

# 3 b
# daca un esantion este memorat pe 4 biti, o ora are 3600s sau 3 600 000ms, si o secunda are 2000 esantioane
# atunci o ora va ocupa 4 * 2 000 * 3 600 / 8 = 3 600 000 bytes
