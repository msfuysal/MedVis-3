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
F_quad = np.fft.fft2(f_quad)
# shifting zero-frequency component to the center of the image
F_quad_shifted = np.fft.fftshift(F_quad)


# plot both f_quad and its fourier transform F_quad
fig, axs = plt.subplots(1, 2, constrained_layout=True)

fig.suptitle("2D Fourier Transform")

# plotting original image in left subplot (f_quad is being plotted in line 13-17; thats why its shown twice)
axs[0].imshow(f_quad, cmap="viridis")
axs[0].set_title("f_quad")

# plotting magnitude of Fourier transformation in right subplot (using log scale for underline clearer the differences)
axs[1].imshow(np.log(1 + np.abs(F_quad_shifted)), cmap="viridis")
axs[1].set_title("|F_quad| (log scale)")

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
    # creating copy of Fourier transform so the "original" is not being influenced
    F_filtered = F.copy()
    # getting number of rows and columns from Fourier transform
    M, N = F.shape
    # creating coordinate grids in the center of the image (0)
    u = np.arange(M) - M//2
    v = np.arange(N) - N//2
    U, V = np.meshgrid(u, v, indexing='ij')
    # defining distance from the center for each frequency component in pixels
    D = np.sqrt(U**2 + V**2)
    # zero out all the high frequencies outside of the threshold radius --> keeping only low frequencies
    F_filtered[D > threshold] = 0  
    return F_filtered

def high_pass_filter(F: np.ndarray, threshold: int) -> np.ndarray:
    # creating copy of Fourier transform so the "original" is not being influenced
    F_filtered = F.copy()
    # getting number of rows and columns from Fourier transform
    M, N = F.shape
     # creating coordinate grids in the center of the image (0)
    u = np.arange(M) - M//2
    v = np.arange(N) - N//2
    U, V = np.meshgrid(u, v, indexing='ij')
    # defining distance from the center for each frequency component in pixels
    D = np.sqrt(U**2 + V**2)
    # zero out all the low frequencies outside of the threshold radius --> keeping only high frequencies
    F_filtered[D < threshold] = 0  
    return F_filtered
 
# apply filters to the fourier transformed F_quad of f_quad
# defining cutoff threshold radius of n pixels --> here: 20 pixels but flexible
threshold_radius = 20
# Applying low-pass and high-pass filters to shifted Fourier transform
F_low = low_pass_filter(F_quad_shifted, threshold_radius)
F_high = high_pass_filter(F_quad_shifted, threshold_radius)

# for both filters: 
# - moving the zero frequency back to top-left corner --> base for inverse FFT
# - compute inverse 2D Fourier transform
# - discarding noise caused by numerical errors
f_low = np.fft.ifft2(np.fft.ifftshift(F_low)).real
f_high = np.fft.ifft2(np.fft.ifftshift(F_high)).real


# plot all results (low pass and high pass filtered f_quad and their fourier transformations)
# fig, grid = plt.subplots(2, 2)
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

fig.suptitle("Fourier Filters")
# plotting the low-pass filtered image in spatial domain
axs[0, 0].imshow(f_low, cmap="viridis")
axs[0, 0].set_title("Low-pass filtered(spatial)")
# plotting the magnitude of low-pass Fourier transform
axs[0, 1].imshow(np.log(1 + np.abs(F_low)), cmap="viridis")
axs[0, 1].set_title("|F_low|")
# plotting the high-pass filtered image in spatial domain
axs[1, 0].imshow(f_high, cmap="viridis")
axs[1, 0].set_title("High-pass filtered(spatial)")
# plotting magnitude of high-pass Fourier transform
axs[1, 1].imshow(np.log(1 + np.abs(F_high)), cmap="viridis")
axs[1, 1].set_title("|F_high|")
plt.show()
# --------------------------------------------------------
