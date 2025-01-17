import scipy
import numpy as np
import warnings
import os
import progressbar

from skimage.filters import median
from skimage import io
from skimage.feature import canny
from skimage.transform import rotate
from skimage.morphology import disk


# Savitzky-Golay filter
def sgolay2d(z, window_size, order, derivative=None):
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
    original = io.imread(folder + "/P" + str(num) + ".tif")

    denoised = median(original, disk(10))

    edges = canny(denoised, sigma=0.1)
    

    # Savitzky-Golay filter
    sgol = sgolay2d(edges, 15, 4)
    mask = sgol < 0.1
    sgol[mask] = 0
    mask = sgol > 0.1
    sgol[mask] = 1

    rotated = rotate(sgol, 270, resize=True)
    height, width = rotated.shape
    x = [0] * 1024

    for i in range(0, height):
        for j in range(0, width):
            if rotated[i, j] == 1:
                x[i] = j
                break

    z = np.polyfit(range(0, 1024), x, fit)
    f = np.poly1d(z)
    x_new = np.linspace(0, 1023, 1024)
    y_new = f(x_new)

    original_rot = rotate(original, 270, resize=True)
    height, width = original_rot.shape

    line = [0] * 1024
    for i in range(height):
        if y_new[i] > x[i]:
            line[i] = y_new[i]
        else:
            line[i] = x[i]

    for i in range(0, height):
        for j in range(0, width):
            original_rot[i, j] = 0
            if int(line[i] + 30) == j:
                break


    for i in range(0, height):
        for j in range(width - 1, 0, -1):
            original_rot[i, j] = 0
            if rotated[i, j] == 1:
                original_rot[i, j - 50:j] = 0
                break

    output = rotate(original_rot, 90, resize=True)
    mask = output < 0.700
    output[mask] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        remove(i, folder, fit, out_folder)
        bar.update(i - num + 1)
    bar.finish()


incrementing_bar()
