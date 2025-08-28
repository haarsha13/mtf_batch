# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pylab as pylab
import numpy as np
import cv2 as cv2
import math as math

from PIL import Image
from scipy import interpolate
from scipy.fft import fft
from enum import Enum
from dataclasses import dataclass


# -------------------------
# Data containers
# -------------------------
@dataclass
class cSet:
    x: np.ndarray
    y: np.ndarray

@dataclass
class cESF:
    rawESF: cSet
    interpESF: cSet
    threshold: float
    width: float
    angle: float
    edgePoly: np.ndarray

@dataclass
class cMTF:
    x: np.ndarray
    y: np.ndarray
    mtfAtNyquist: float
    width: float

class Verbosity(Enum):
    NONE = 0
    BRIEF = 1
    DETAIL = 2


# -------------------------
# Image helpers
# -------------------------
class Transform:

    @staticmethod
    def _raw_edge_angle_deg(arr01):
        a = np.asarray(arr01, dtype=float)
        if a.max() > 1.0:
            a = a / 255.0
        a = np.squeeze(a)
        if a.ndim == 3 and a.shape[2] >= 3:
            a = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
        edgeImg = cv2.Canny(np.uint8(np.clip(a,0,1)*255), 40, 90, L2gradient=True)
        line = np.argwhere(edgeImg == 255)
        if line.size < 2:
            return 0.0
        edgePoly = np.polyfit(line[:,1], line[:,0], 1)
        angle = math.degrees(math.atan(-edgePoly[0]))
        return float(angle)

    @staticmethod
    def _otsu_threshold01(gray):
        g = np.clip(gray, 0.0, 1.0)
        hist, bin_edges = np.histogram(g.ravel(), bins=256, range=(0.0, 1.0))
        prob = hist.astype(float) / (hist.sum() + 1e-12)
        omega = np.cumsum(prob)
        centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
        mu = np.cumsum(prob * centers)
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
        idx = int(np.nanargmax(sigma_b2))
        return float(bin_edges[idx])

    @staticmethod
    def _michelson_contrast01(gray):
        t = Transform._otsu_threshold01(gray)
        dark = gray[gray <= t]
        bright = gray[gray > t]
        if dark.size < 5 or bright.size < 5:
            return 0.0
        Imin = float(dark.mean())
        Imax = float(bright.mean())
        denom = (Imax + Imin)
        if abs(denom) < 1e-12:
            return 0.0
        return max(0.0, (Imax - Imin) / denom)

    @staticmethod
    def LoadImg(file):
        """Return (PIL_gray_img, width, height)."""
        img = Image.open(file)
        SHAPE_x, SHAPE_y = img.size
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
            gsimg = img
        else:
            gsimg = img.convert('L')
        return gsimg, SHAPE_x, SHAPE_y

    @staticmethod
    def Arrayify(img):
        """Accept a PIL image OR a (img, w, h) tuple and return 0..1 float array."""
        if isinstance(img, tuple):
            img = img[0]
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
            arr = np.asarray(img, dtype=np.float64) / 65535.0
        else:
            arr = np.asarray(img, dtype=np.float64) / 255.0
        return arr

    @staticmethod
    def Imagify(Arr):
        return Image.fromarray(np.uint8(np.clip(Arr,0,1)*255), mode='L')

    @staticmethod
    def Orientify(Arr):
        tl = np.average(Arr[0:2, 0:2])
        tr = np.average(Arr[0:2, -3:-1])
        bl = np.average(Arr[-3:-1, 0:2])
        br = np.average(Arr[-3:-1, -3:-1])

        corners = [tl, tr, bl, br]
        cornerIndexes = np.argsort(corners)

        if (cornerIndexes[0] + cornerIndexes[1]) == 1:
            vertical = 1
        elif (cornerIndexes[0] + cornerIndexes[1]) == 5:
            Arr = np.flip(Arr, axis=0)
            vertical = 1
        elif (cornerIndexes[0] + cornerIndexes[1]) == 2:
            Arr = np.transpose(Arr)
            vertical = 0
        elif (cornerIndexes[0] + cornerIndexes[1]) == 4:
            Arr = np.flip(np.transpose(Arr), axis=0)
            vertical = 0

        return Arr, vertical


# -------------------------
# MTF pipeline
# -------------------------
class MTF:
    @staticmethod
    def crop(values, distances, head, tail):
        isIncrementing = True
        if distances[0] > distances[-1]:
            isIncrementing = False
            distances = -distances
            dummy = -tail
            tail = -head
            head = dummy

        hindex = (np.where(distances < head)[0])
        tindex = (np.where(distances > tail)[0])

        if hindex.size < 2:
            h = 0
        else:
            h = np.amax(hindex)
        if tindex.size == 0:
            t = distances.size
        else:
            t = np.amin(tindex)

        if not isIncrementing:
            distances = -distances

        return cSet(distances[h:t], values[h:t])

    @staticmethod
    def GetESF(Arr, edgePoly, verbose=Verbosity.NONE):
        Y, X = Arr.shape[0], Arr.shape[1]
        values = np.reshape(Arr, X*Y)

        distance = np.zeros((Y,X))
        column = np.arange(0,X) + 0.5
        for i in range(Y):
            distance[i,:] = (edgePoly[0]*column - (i+0.5) + edgePoly[1]) / np.sqrt(edgePoly[0]*edgePoly[0] + 1)

        distances = np.reshape(distance, X*Y)
        indexes = np.argsort(distances)

        sign = 1
        if np.average(values[indexes[:10]]) > np.average(values[indexes[-10:]]):
            sign = -1

        values = values[indexes]
        distances = sign*distances[indexes]

        if distances[0] > distances[-1]:
            distances = np.flip(distances)
            values = np.flip(values)

        if (verbose == Verbosity.BRIEF):
            print("Raw ESF [done] (Distance from {0:2.2f} to {1:2.2f})"
                  .format(sign*distances[0], sign*distances[-1]))
        elif (verbose == Verbosity.DETAIL):
            x = [0, np.size(Arr,1)-1]
            y = np.polyval(edgePoly, x)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Raw ESF')
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.imshow(Arr, cmap='gray')
            ax1.plot(x, y, color='r')
            ax2.plot(distances, values)
            plt.show(block=False); plt.show()

        return cSet(distances, values)

    @staticmethod
    def GetESF_crop(Arr, verbose=Verbosity.NONE):
        imgArr, verticality = Transform.Orientify(Arr)
        edgeImg = cv2.Canny(np.uint8(np.clip(imgArr,0,1)*255), 40, 90, L2gradient=True)

        line = np.argwhere(edgeImg == 255)
        if line.size < 2:
            raise RuntimeError("No slanted edge detected — choose a clearer edge patch.")
        edgePoly = np.polyfit(line[:,1], line[:,0], 1)
        angle = math.degrees(math.atan(-edgePoly[0]))

        finalEdgePoly = edgePoly.copy()
        if angle > 0:
            imgArr = np.flip(imgArr, axis=1)
            finalEdgePoly[1] = np.polyval(edgePoly, np.size(imgArr, 1)-1)
            finalEdgePoly[0] = -edgePoly[0]

        esf = MTF.GetESF(imgArr, finalEdgePoly, Verbosity.BRIEF)

        esf_Values = esf.y
        esf_Distances = esf.x

        vmax = np.amax(esf_Values)
        vmin = np.amin(esf_Values)
        threshold = (vmax - vmin) * 0.1

        head = np.amax(esf_Distances[(np.where(esf_Values < vmin + threshold))[0]])
        tail = np.amin(esf_Distances[(np.where(esf_Values > vmax - threshold))[0]])

        width = abs(head - tail)
        esfRaw = MTF.crop(esf_Values, esf_Distances, head - 1.2*width, tail + 1.2*width)

        qs = np.linspace(0,1,20)[1:-1]
        knots = np.quantile(esfRaw.x, qs)
        tck = interpolate.splrep(esfRaw.x, esfRaw.y, t=knots, k=3)
        ysmooth = interpolate.splev(esfRaw.x, tck)

        InterpDistances = np.linspace(esfRaw.x[0], esfRaw.x[-1], 500)
        InterpValues = np.interp(InterpDistances, esfRaw.x, ysmooth)

        esfInterp = cSet(InterpDistances, InterpValues)

        if (verbose == Verbosity.BRIEF):
            print("ESF Crop [done] (Distance from {0:2.2f} to {1:2.2f})"
                  .format(esfRaw.x[0], esfRaw.x[-1]))
        elif (verbose == Verbosity.DETAIL):
            x = [0, np.size(imgArr,1)-1]
            y = np.polyval(finalEdgePoly, x)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('ESF Crop')
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax2.plot(esfRaw.x, esfRaw.y, InterpDistances, InterpValues)
            plt.show(block=False); plt.show()

        return cESF(esfRaw, esfInterp, threshold, width, angle, edgePoly)

    @staticmethod
    def GetLSF(ESF,
               normalize=True,
               verbose=Verbosity.NONE,
               window: str | None = "tukey",
               alpha: float = 0.25,
               tails_frac: float = 0.10,
               pad_to: int | None = None):
        """Differentiate ESF -> LSF, baseline-correct, window, optionally zero-pad."""
        # derivative
        lsfDividend = np.diff(ESF.y)
        lsfDivisor  = np.diff(ESF.x)
        lsfDivisor  = np.where(np.abs(lsfDivisor) < 1e-12, 1.0, lsfDivisor)
        lsfValues   = np.divide(lsfDividend, lsfDivisor)
        lsfDistances = ESF.x[:-1]

        # baseline from tails
        n = lsfValues.size
        k = max(1, int(tails_frac * n))
        tails_mean = 0.5 * (np.mean(lsfValues[:k]) + np.mean(lsfValues[-k:]))
        lsfValues = lsfValues - tails_mean

        # window
        if window is not None:
            wl = window.lower()
            if wl == "tukey":
                w = np.ones(n)
                if alpha > 0:
                    x = np.linspace(0, 1, n)
                    first  = x < (alpha/2)
                    middle = (x >= (alpha/2)) & (x <= (1 - alpha/2))
                    last   = x > (1 - alpha/2)
                    w[first]  = 0.5 * (1 + np.cos(2*np.pi*(x[first]/alpha - 0.5)))
                    w[last]   = 0.5 * (1 + np.cos(2*np.pi*((x[last]-1)/alpha + 0.5)))
                    w[middle] = 1.0
            elif wl == "hann":
                w = 0.5 * (1 - np.cos(2*np.pi*np.arange(n)/(n-1)))
            elif wl == "hamming":
                w = 0.54 - 0.46*np.cos(2*np.pi*np.arange(n)/(n-1))
            else:
                w = np.ones(n)
            lsfValues = lsfValues * w

        # area normalise (MTF(0) ≈ 1)
        area = np.trapz(lsfValues, lsfDistances)
        if np.abs(area) < 1e-12:
            area = 1.0
        lsfValues = lsfValues / area

        # zero-padding
        if pad_to is not None and pad_to > n:
            pad_n = pad_to - n
            lsfValues = np.pad(lsfValues, (0, pad_n), mode='constant', constant_values=0)
            dx = (lsfDistances[-1] - lsfDistances[0]) / max(n-1,1)
            extra_x = lsfDistances[-1] + dx*np.arange(1, pad_n+1)
            lsfDistances = np.concatenate([lsfDistances, extra_x])

        if (verbose == Verbosity.BRIEF):
            print("LSF [done] (window={}, pad_to={})".format(window, pad_to))
        elif (verbose == Verbosity.DETAIL):
            fig, ax1 = plt.subplots(1)
            ax1.plot(lsfDistances, lsfValues)
            ax1.set_title("LSF (window={}, pad_to={})".format(window, pad_to))
            ax1.grid(True); ax1.minorticks_on()
            plt.show(block=False); plt.show()

        return cSet(lsfDistances, lsfValues)

    @staticmethod
    def GetMTF(LSF, fraction, verbose=Verbosity.NONE):
        """FFT LSF -> MTF, normalise frequency axis to 0..1 (Nyquist=0.5)."""
        N = LSF.x.size
        dx = (LSF.x[-1] - LSF.x[0]) / max(N-1, 1)
        if abs(dx) < 1e-12:
            dx = 1.0

        spec  = np.abs(fft(LSF.y))
        freqs = np.fft.fftfreq(N, d=dx)

        keep = freqs >= 0
        freqs = freqs[keep]
        spec  = spec[keep]

        # normalise frequency axis to [0,1]
        if freqs.max() > 0:
            norm_f = (freqs / freqs.max())
        else:
            norm_f = freqs

        grid = np.linspace(0.0, 1.0, 200)
        interp = interpolate.interp1d(norm_f, spec, kind='cubic',
                                      bounds_error=False, fill_value="extrapolate")
        interpValues = interp(grid)

        mtf_at_nyq = float(np.squeeze(interp(0.5)))
        valueAtNyquist = mtf_at_nyq * 100.0

        crossing_idx = np.where(interpValues <= fraction)[0]
        if len(crossing_idx) > 0 and crossing_idx[0] > 0:
            i = crossing_idx[0]
            x0, y0 = grid[i-1], interpValues[i-1]
            x1, y1 = grid[i],   interpValues[i]
            cutoff_freq = x1 if np.abs(y1-y0) < 1e-12 else x0 + (fraction - y0)*(x1 - x0)/(y1 - y0)
        else:
            cutoff_freq = None

        if (verbose == Verbosity.BRIEF):
            print("MTF [done]")
        elif (verbose == Verbosity.DETAIL):
            fig, ax1 = plt.subplots(1)
            ax1.plot(grid, interpValues)
            ax1.set_xlabel("Normalized Frequency")
            ax1.set_ylabel("MTF Value")
            ax1.set_title("MTF ({:.3f} at Nyquist)".format(mtf_at_nyq))
            ax1.grid(True); ax1.minorticks_on()
            plt.show(block=False); plt.show()

        return cMTF(grid, interpValues, valueAtNyquist, -1.0), cutoff_freq

    @staticmethod
    def MTF_Full(imgArr,
                 fraction,
                 w=None, h=None,
                 verbose=Verbosity.NONE,
                 # expose windowing knobs:
                 window: str | None = "tukey",
                 alpha: float = 0.25,
                 tails_frac: float = 0.10,
                 pad_to: int | None = 4096):
        """End-to-end: ESF -> LSF (with windowing) -> MTF."""
        contrast = Transform._michelson_contrast01(imgArr)
        imgArr, verticality = Transform.Orientify(imgArr)
        w, h = imgArr.shape[0], imgArr.shape[1]
        esf = MTF.GetESF_crop(imgArr, Verbosity.DETAIL)     # raw+interp ESF
        lsf = MTF.GetLSF(esf.interpESF, True, Verbosity.DETAIL,
                         window=window, alpha=alpha, tails_frac=tails_frac, pad_to=pad_to)
        mtf, cutoff_freq = MTF.GetMTF(lsf, fraction, Verbosity.DETAIL)

        verticality = "Vertical" if verticality > 0 else "Horizontal"

        if (verbose == Verbosity.DETAIL):
            plt.figure(figsize=(8,6))
            x = [0, np.size(imgArr,1)-1]
            y = np.polyval(esf.edgePoly, x)

            gs = plt.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = plt.subplot(gs[:, 1])

            # Image + edge
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax1.axis('off')
            ax1.set_title(f"Image Dimensions: {w} by {h}\nEdge Profile: {verticality}")

            # ESF (raw + interp)
            ax2.plot(esf.rawESF.x, esf.rawESF.y,
                     esf.interpESF.x, esf.interpESF.y)
            top = np.max(esf.rawESF.y)-esf.threshold
            bot = np.min(esf.rawESF.y)+esf.threshold
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [top, top], color='red')
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [bot, bot], color='red')
            ax2.grid(True); ax2.minorticks_on()

            # LSF (windowed)
            ax3.plot(lsf.x, lsf.y)
            ax3.grid(True); ax3.minorticks_on()

            # MTF
            ax4.plot(mtf.x, mtf.y)
            nyq_val = mtf.mtfAtNyquist/100.0
            ax4.set_title(f"MTF{int(fraction*100)}: {cutoff_freq if cutoff_freq is not None else float('nan'):0.3f}\n"
                          f"MTF at Nyquist: {nyq_val:0.5f}")
            ax4.plot(0.5, nyq_val, 'o', color='red', linestyle='None', label='Nyquist', ms=3)
            if cutoff_freq is not None:
                ax4.plot(cutoff_freq, fraction, 'o', color='red', linestyle='None',
                         label=f'MTF{fraction*100}', ms=3)
            ax4.text(0.5, 0.99, f"Angle: {esf.angle:0.3f}°", ha='left', va='top')
            ax4.text(0.5, 0.94, f"Width: {esf.width:0.3f} px", ha='left', va='top')
            ax4.text(0.5, 0.89, f"Threshold: {esf.threshold:0.3f}", ha='left', va='top')
            ax4.text(0.5, 0.84, f"Contrast: {contrast*100:0.1f}%", ha='left', va='top')
            ax4.set_xlabel('Normalized Frequency')
            ax4.set_ylabel('MTF Value')
            ax4.minorticks_on()
            plt.tight_layout()

        return cMTF(mtf.x, mtf.y, mtf.mtfAtNyquist, esf.width)

