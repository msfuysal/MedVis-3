"""Insert your solutions for exercise 3 here."""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# define the 2D function
f_quad = np.zeros((128, 128))
S = np.arange(31, 95)
f_quad[np.ix_(S, S)] = 1

# demonstrate plotting
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.set_title("f_quad")
ax.imshow(f_quad, cmap="viridis")
fig.suptitle("Plotting Example")
plt.show()
# --------------------------------------------------------

# --------------------------------------------------------
# Insert solutions for 3a here

# apply the 2D fourier transform to f_quad


# plot both f_quad and its fourier transform F_quad
fig, axs = plt.subplots(1, 2, constrained_layout=True)

fig.suptitle("2D Fourier Transform")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 3b here
def box_filter_2d(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ...

# define gauss and laplace kernels
gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
laplace_kernel = np.array([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]])

# apply low pass and high pass to f_quad


# apply the 2D Fourier transform to the results


# plot all results (low pass and high pass filtered f_quad and their fourier transformations)
fig, axs = plt.subplots(2, 2, constrained_layout=True)

fig.suptitle("Low-pass and High-pass")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 3c here
def low_pass_filter(F: np.ndarray, threshold: int) -> np.ndarray:
    ...

def high_pass_filter(F: np.ndarray, threshold: int) -> np.ndarray:
    ...

# apply filters to the fourier transformed F_quad of f_quad


# plot all results (low pass and high pass filtered f_quad and their fourier transformations)
fig, grid = plt.subplots(2, 2)

fig.suptitle("Fourier Filters")
plt.show()
# --------------------------------------------------------
