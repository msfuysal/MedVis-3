"""Insert your solutions for exercise 2 here."""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# sampling details of the function f
sf = 1000  # sampling frequency
T = 1 / sf  # sampling period
N = 1200  # number of samples
x = np.arange(N) * T  # time vector

# vector containing the frequencies for plotting in fourier space
# we can shift the frequencies using fftshift so that 0 is in the middle of the plot
# [-500., -499.16666667, ..., 0, ...,  498.33333333,  499.16666667]
freq = np.fft.fftshift(np.fft.fftfreq(N, d=T))

# create a function f as a superposition of two cosine waves
f = 0.5 * np.cos(2 * np.pi * 30 * x) + 0.8 * np.cos(2 * np.pi * 80 * x)

# create noisy version of f for exercise 2c
noise = np.random.randn(N)
f_noisy = f + noise

# plot both functions to demonstrate the plotting
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
axs[0].set_title("f")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].plot(x, f, "r")
axs[1].set_title("f with noise")
axs[1].set_xlabel("x")
axs[1].set_ylabel("$f_{noisy}(x)$")
axs[1].plot(x, f_noisy, "b")
fig.suptitle("Plotting Example")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2a and 2b here
def fourier_transform(f: np.ndarray, inv: bool = False) -> np.ndarray:
    ...

# use the function to get the fourier transform F of the function f


# plot f and F
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("Fourier Transform")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2c here
def box_filter(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ...

# define the kernel of the box filter
kernel = 1 / 4 * np.array([1, 2, 1])

# filter f_noisy using the box filter


# plot filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("Box Filter")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2d here
def ft_denoise(F: np.ndarray, threshold: int) -> np.ndarray:
    ...

# fourier transform f_noisy, apply fourier filter and apply inverse fourier transformation


# plot the fourier filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("FT Filter")
plt.show()
# --------------------------------------------------------
