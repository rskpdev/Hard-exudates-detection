import scipy
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import progressbar

from oct2py import octave
from skimage.filters import median, sobel_h
from skimage import io
from skimage.feature import canny
from skimage.transform import rotate
from skimage.morphology import disk
from scipy.signal import savgol_filter
from scipy import ndimage as ndi
from skimage.exposure import equalize_hist


def sgolay2d(z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - \
        np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + \
        np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - \
        np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + \
        np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - \
        np.abs(
            np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + \
        np.abs(
            np.flipud(np.fliplr(z[-half_size - 1:-1,
                      -half_size - 1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - \
        np.abs(
            np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - \
        np.abs(
            np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'),
        scipy.signal.fftconvolve(Z, -c, mode='valid')


def remove(num, folder, fit, out_folder):
    name = folder + "/P" + str(num) + ".tif"
    input1 = io.imread(name)
    input1_copy = input1
    original = octave.shadow_remove(name)
    denoised = median(original, disk(20))
    edges = sobel_h(denoised)
    sgol = sgolay2d(edges, 15, 4)
    height, width = sgol.shape
    mask = sgol < 0.019
    sgol[mask] = 0
    mask = sgol > 0.019
    sgol[mask] = 1
    sgol[800:, :] = 0

    x = [0] * width

    for i in range(width):
        for j in range(height - 1, 0, -1):
            if sgol[j, i] == 1:
                x[i] = j
                break
    z = np.polyfit(range(0, 1024), x, fit)
    f = np.poly1d(z)
    x_new = np.linspace(0, 1023, 1024)
    y_new = f(x_new)
    clear = max(y_new)

    sgol[int(clear):, :] = 0

    x = [0] * width

    for i in range(width):
        for j in range(height - 1, 0, -1):
            if sgol[j, i] == 1:
                x[i] = j
                break
    z = np.polyfit(range(0, 1024), x, fit)
    f = np.poly1d(z)
    x_new = np.linspace(0, 1023, 1024)
    y_new = f(x_new)

    for i in range(width - 1):
        if any(t == 1 for t in sgol[x[i] - 16:x[i] + 16, i + 1]):
            pass
        else:
            x[i + 1] = int(y_new[i + 1])

    for i in range(width):
        input1[x[i] - 15:, i] = 0

    denoised = median(input1, disk(10))
    edges = canny(denoised, sigma=0.1)

    sgol = sgolay2d(edges, 15, 4)
    mask = sgol < 0.1
    sgol[mask] = 0
    mask = sgol > 0.1
    sgol[mask] = 1

    for i in range(0, width):
        for j in range(0, height):
            input1[j, i] = 0
            if sgol[j, i] == 1:
                input1[j:j + 50, i] = 0
                break

    output = equalize_hist(input1_copy)
    # output = median(output, disk(5))
    # mask = input1 == 0
    # output[mask] = 0
    # mask = output < 170
    # output[mask] = 0
    io.imsave(out_folder + "/" + str(num) + ".bmp", output)


def incrementing_bar():
    folder = input("enter input folder name: ")
    out_folder = input("enter output folder name: ")
    num = int(input("enter starting number: "))
    fit = int(input("enter degree of polynomial fit: "))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    bar = progressbar.ProgressBar(widgets=[
        progressbar.Percentage(),
        progressbar.Bar(),
    ], max_value=256).start()
    for i in range(num, num + 256):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            remove(i, folder, fit, out_folder)
        bar.update(i - num + 1)
    bar.finish()

incrementing_bar()
