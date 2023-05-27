"""
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com
"""
import numpy as np
import math
from skimage import color, filters


def calculate_uw_metrics(a):
    rgb = a
    gray = color.rgb2gray(a)

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int(al1 * len(rgl))
    T2 = np.int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    r_sobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    g_sobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    b_sobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    r_sobel = np.round(r_sobel).astype(np.uint8)
    g_sobel = np.round(g_sobel).astype(np.uint8)
    b_sobel = np.round(b_sobel).astype(np.uint8)

    r_eme = eme(r_sobel)
    g_eme = eme(g_sobel)
    b_eme = eme(b_sobel)

    uism = 0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm

    return uicm, uism, uiconm, uiqm


def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2.0 / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):
            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            # new version
            if blockmin == 0:
                blockmin += 1
            if blockmax == 0:
                blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma) ** c


def logamee(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1.0 / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):
            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if top == 0.0 or bottom == 0.0:
                s += 0
            else:
                m = top / bottom
                s += (m) * np.log(m)

    return plipmult(w, s)
