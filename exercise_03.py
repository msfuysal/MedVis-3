"""Insert your solutions for exercise 3 here."""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# define the 2D function
f_quad = np.zeros((128, 128))
S = np.arange(31, 95)
f_quad[np.ix_(S, S)] = 1

# Plot the function:
plt.imshow(f_quad)


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

# plotting original image in left subplot (f_quad is being plotted in line 13-17; thats why its shown twice)
axs[0].imshow(f_quad, cmap="viridis")
axs[0].set_title("f_quad")

# plotting magnitude of Fourier transformation in right subplot (using log scale for underline clearer the differences)
axs[1].imshow(np.log(1 + np.abs(F_Quad_shifted)), cmap="viridis")
axs[1].set_title("|F_quad| (log scale)")

plt.show()
# --------------------------------------------------------
# --------------------------------------------------------
# Insert solutions for 3b here
def box_filter_2d(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """This function implements a filter by using a kernel"""
    padded_f = np.pad(f, ((1, 1), (1, 1)), mode='constant')
    filtered_f = np.zeros_like(f)
    for i in range(f.shape[0]): #apply to each pixel
        for j in range(f.shape[1]):
            region = padded_f[i:i+3, j:j+3]
            filtered_f[i, j] = np.sum(region * kernel)
    return filtered_f

# define gauss and laplace kernels
gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
laplace_kernel = np.array([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]])

# apply low pass and high pass to f_quad
lowpass_filtered = box_filter_2d(f_quad, gauss_kernel)
highpass_filtered = box_filter_2d(f_quad, laplace_kernel)

# apply the 2D Fourier transform to the results
F_lowpass = np.fft.fft2(lowpass_filtered)
F_lowpass_shifted = np.fft.fftshift(F_lowpass)
F_highpass = np.fft.fft2(highpass_filtered)
F_highpass_shifted = np.fft.fftshift(F_highpass)

# plot all results (lowpass and highpass filtered f_Quad and their fourier transformations)
fig, axs = plt.subplots(2, 2, constrained_layout=True)

# low-pass result
axs[0, 0].imshow(lowpass_filtered, cmap="gray")
axs[0, 0].set_title("Low-pass filtered f_quad")



# low-pass Fourier transform
axs[0, 1].imshow(np.log(1 + np.abs(F_lowpass_shifted)), cmap="gray")
axs[0, 1].set_title("Fourier lowpass filtered")


# high-pass result
axs[1, 0].imshow(highpass_filtered, cmap="gray")
axs[1, 0].set_title("Highpass filtered f_Quad")


# high-pass Fourier transform
axs[1, 1].imshow(np.log(1 + np.abs(F_highpass_shifted)), cmap="gray")
axs[1, 1].set_title("Fourier Highpass")


fig.suptitle("Lowpass and Highpass")
plt.show()

conv_result = lowpass_filtered
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

"""
Comparison of Task 3b (spatial filtering) and Task 3c (frequency filtering):

The original image `f_quad` is basically a square, so it has sharp edges. Because of that, its Fourier transform (see Plot 2) shows strong high-frequency components — especially those cross-shaped lines in the middle.

In Task 3b, we used spatial filters like a Gaussian (for low-pass) and something similar to Laplace (for high-pass). The results (Plot 3) looked pretty natural:
- The low-pass filter smoothed the image nicely.
- The high-pass filter brought out the edges clearly without adding noise.

In Task 3c, we applied ideal filters in the frequency domain. These filters are much more “aggressive”:
- The ideal low-pass filter caused ringing effects (Gibbs effect) around the edges.
- The ideal high-pass filter made the image sharper, but also added some noise.

To sum it up:
- Task 3b (spatial filters) gave smoother and more natural results.
- Task 3c (frequency filters) gave more control, but can create artifacts if not used carefully.
- It depends on the goal — if you want clean visuals, go for 3b; if you need precise frequency editing, 3c is better.
"""