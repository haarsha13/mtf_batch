# -*- coding: utf-8 -*-
"""Resolution and Sharpness of Images (Fixed)

Edits made to prevent runtime errors and improve robustness.

Original authors: Damian Howe and Haarsha Krishna
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import math as math

from PIL import Image, ImageOps
from scipy import interpolate
from scipy.fft import fft
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class cSet:
    x: np.ndarray  # storing indexes as a numpy array
    y: np.ndarray  # storing values as a numpy array

@dataclass
class cESF:
    rawESF: cSet            # raw ESF data as numpy arrays
    interpESF: cSet         # interpolated ESF data as numpy arrays
    threshold: float        # threshold used for ESF/MTF
    width: float            # pixel transition size
    angle: float            # slant angle (deg)
    edgePoly: np.ndarray    # polynomial of slant

@dataclass
class cMTF:
    x: np.ndarray           # array for indexes
    y: np.ndarray           # array for values
    mtfAtNyquist: float     # MTF value at sampling speed limit (Nyquist Freq) (percent)
    width: float            # pixel transition size

@dataclass
class Verbosity(Enum):  # output types/level of output
    NONE = 0   # none
    BRIEF = 1  # text
    DETAIL = 2 # graphical

# -----------------------------
# Utility functions
# -----------------------------

def _otsu_threshold01(gray: np.ndarray) -> float:
    """Return Otsu threshold for a 0..1 grayscale array."""
    import numpy as _np
    g = _np.clip(gray, 0.0, 1.0)
    hist, bin_edges = _np.histogram(g.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(float)
    prob = hist / (hist.sum() + 1e-12)
    omega = _np.cumsum(prob)
    centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
    mu = _np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(_np.nanargmax(sigma_b2))
    return float(bin_edges[idx])

def _michelson_contrast01(gray: np.ndarray) -> float:
    """Return Michelson contrast (0..1) for a 0..1 grayscale array using Otsu split."""
    import numpy as _np
    g = _np.clip(gray, 0.0, 1.0)
    t = _otsu_threshold01(g)
    dark = g[g <= t]
    bright = g[g > t]
    if dark.size < 5 or bright.size < 5:
        return 0.0
    Imin = float(dark.mean())
    Imax = float(bright.mean())
    denom = (Imax + Imin)
    if abs(denom) < 1e-12:
        return 0.0
    return max(0.0, (Imax - Imin) / denom)

# -----------------------------
# Transform helpers
# -----------------------------

class Transform:

    @staticmethod
    def LoadImg(file):
        img = Image.open(file)
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:  # 16-bit
            gsimg = img
        else:
            gsimg = img.convert('L')  # 8-bit grayscale
        return gsimg

    @staticmethod
    def Arrayify(img: Image.Image) -> np.ndarray:
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
            arr = np.asarray(img, dtype=np.double) / 65535.0  # normalize
        else:
            arr = np.asarray(img, dtype=np.double) / 255.0    # normalize
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def Imagify(Arr: np.ndarray) -> Image.Image:
        a = np.clip(Arr, 0.0, 1.0)
        a8 = (a * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(a8, mode='L')
        return img

    @staticmethod
    def Orientify(Arr: np.ndarray):
        """Orient the array to a canonical orientation and return (Arr, vertical_flag)."""
        tl = np.average(Arr[0:2, 0:2])
        tr = np.average(Arr[0:2, -3:-1])
        bl = np.average(Arr[-3:-1, 0:2])
        br = np.average(Arr[-3:-1, -3:-1])

        corners = [tl, tr, bl, br]
        cornerIndexes = np.argsort(corners)

        # vertical = 1 for vertical, 0 for horizontal
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
        else:
            vertical = 1  # safe default
        return Arr, vertical  # returns the array and whether it is vertical or not

# -----------------------------
# MTF functions
# -----------------------------

class MTF:
    # distances = microns/pixels away from the slant edge
    # values = array of brightnesses
    # head = top limit
    # tail = bottom limit
    @staticmethod
    def crop(values, distances, head, tail):
        isIncrementing = True
        distances = np.asarray(distances).copy()
        values = np.asarray(values).copy()
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

        if isIncrementing == False:
            distances = -distances

        return cSet(distances[h:t], values[h:t])

    @staticmethod
    def GetESF(Arr, edgePoly, verbose=Verbosity.NONE):
        Y = Arr.shape[0]
        X = Arr.shape[1]

        values = np.reshape(Arr, X * Y)

        distance = np.zeros((Y, X))
        column = np.arange(0, X) + 0.5
        for i in range(Y):
            distance[i, :] = (edgePoly[0]*column - (i+0.5) + edgePoly[1]) / np.sqrt(edgePoly[0]*edgePoly[0] + 1)

        distances = np.reshape(distance, X * Y)
        indexes = np.argsort(distances)

        sign = 1
        if np.average(values[indexes[:10]]) > np.average(values[indexes[-10:]]):
            sign = -1

        values = values[indexes]
        distances = sign * distances[indexes]

        if (distances[0] > distances[-1]):
            distances = np.flip(distances)
            values = np.flip(values)

        if (verbose == Verbosity.BRIEF):
            print("Raw ESF [done] (Distance from {:.2f} to {:.2f})".format(sign*distances[0], sign*distances[-1]))

        elif (verbose == Verbosity.DETAIL):
            x = [0, np.size(Arr, 1)-1]
            y = np.polyval(edgePoly, x)

            fig = plt.gcf()
            try:
                fig.canvas.manager.set_window_title('Raw ESF')
            except Exception:
                pass
            ax1, ax2 = plt.subplots(2)
            ax1.imshow(Arr, cmap='gray')
            ax1.plot(x, y, color='r')
            ax2.plot(distances, values)
            plt.show(block=False)

        return cSet(distances, values)

    @staticmethod
    def GetESF_crop(Arr, verbose=Verbosity.NONE):
        imgArr, vertical = Transform.Orientify(Arr)  # FIX: unpack orientify
        edgeImg = cv2.Canny(np.uint8(np.clip(imgArr,0,1)*255), 40, 90, L2gradient=True)

        line = np.argwhere(edgeImg == 255)
        if line.size < 2:
            # Fallback: dummy edge
            edgePoly = np.array([0.0, imgArr.shape[0]//2], dtype=float)
        else:
            edgePoly = np.polyfit(line[:, 1], line[:, 0], 1)
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

        qs = np.linspace(0, 1, 20)[1:-1]
        knots = np.quantile(esfRaw.x, qs)
        tck = interpolate.splrep(esfRaw.x, esfRaw.y, t=knots, k=3)
        ysmooth = interpolate.splev(esfRaw.x, tck)

        InterpDistances = np.linspace(esfRaw.x[0], esfRaw.x[-1], 500)
        InterpValues = np.interp(InterpDistances, esfRaw.x, ysmooth)

        esfInterp = cSet(InterpDistances, InterpValues)

        if (verbose == Verbosity.BRIEF):
            print("ESF Crop [done] (Distance from {:.2f} to {:.2f})".format(esfRaw.x[0], esfRaw.x[-1]))

        elif (verbose == Verbosity.DETAIL):
            x = [0, np.size(imgArr, 1)-1]
            y = np.polyval(finalEdgePoly, x)

            fig = plt.gcf()
            try:
                fig.canvas.manager.set_window_title('ESF Crop')
            except Exception:
                pass
            ax1, ax2 = plt.subplots(2)
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax2.plot(esfRaw.x, esfRaw.y, InterpDistances, InterpValues)
            plt.show(block=False)

        return cESF(esfRaw, esfInterp, threshold, width, angle, finalEdgePoly)

    @staticmethod
    def Simplify_ESF(ESF, verbose=Verbosity.NONE):
        res = np.unique(ESF.x, return_index=True, return_counts=True)

        indexes = res[1]
        counts = res[2]
        sz = np.size(res[0])

        distances = ESF.x[indexes]
        values = np.zeros(sz, dtype=np.float64)

        for i in range(sz):
            values[i] = np.sum(ESF.y[indexes[i]:indexes[i]+counts[i]]) / counts[i]

        if (verbose == Verbosity.BRIEF):
            print("ESF Simplification [done] (Size from {0:d} to {1:d})".format(np.size(ESF.x), np.size(distances)))

        elif (verbose == Verbosity.DETAIL):
            fig = plt.gcf()
            try:
                fig.canvas.manager.set_window_title("ESF Simplification")
            except Exception:
                pass
            ax1, ax2 = plt.subplots(2)
            ax1.plot(ESF.x, ESF.y)
            ax2.plot(distances, values)
            plt.show(block=False)

        return cSet(distances, values)

    @staticmethod
    def GetLSF(ESF, normalize=True, verbose=Verbosity.NONE):
        lsfDividend = np.diff(ESF.y)
        lsfDivisor = np.diff(ESF.x)

        with np.errstate(divide='ignore', invalid='ignore'):
            lsfValues = np.divide(lsfDividend, lsfDivisor, out=np.zeros_like(lsfDividend), where=lsfDivisor!=0)
        lsfDistances = ESF.x[0:-1]

        if normalize:
            peak = np.max(np.abs(lsfValues)) if lsfValues.size else 1.0
            if peak < 1e-12:
                peak = 1.0
            lsfValues = lsfValues / peak

        if (verbose == Verbosity.BRIEF):
            print("LSF [done]")

        elif (verbose == Verbosity.DETAIL):
            fig = plt.gcf()
            try:
                fig.canvas.manager.set_window_title("LSF")
            except Exception:
                pass
            (ax1,) = plt.subplots(1)
            ax1.plot(lsfDistances, lsfValues)
            plt.show(block=False)
        return cSet(lsfDistances, lsfValues)

    @staticmethod
    def GetMTF(LSF, fraction, verbose=Verbosity.NONE):
        """Return (cMTF, cutoff_freq) where cutoff is the first freq where MTF<=fraction."""
        N = int(np.size(LSF.x))
        if N < 2:
            return cMTF(np.array([0.0]), np.array([1.0]), 0.0, -1.0), None

        px = N / (LSF.x[-1] - LSF.x[0] + 1e-12)  # samples per pixel-distance
        denom = float(np.sum(LSF.y))
        if abs(denom) < 1e-12:
            denom = 1e-12
        values = (1.0/denom) * np.abs(fft(LSF.y))
        distances = np.arange(0, N) / N * px  # frequency axis

        # Interpolate onto 0..1 normalized frequency
        interpDistances = np.linspace(0.0, 1.0, 200)
        interp = interpolate.interp1d(distances, values, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interpValues = interp(interpDistances)

        # Nyquist at 0.5 on the normalized axis
        nyq_idx = np.argmin(np.abs(interpDistances - 0.5))
        valueAtNyquist = float(interpValues[nyq_idx]) * 100.0  # percent

        # Find first crossing <= fraction
        target = float(fraction)
        crossing_idx = np.where(interpValues <= target)[0]
        if len(crossing_idx) > 0:
            cutoff_freq = float(interpDistances[crossing_idx[0]])
        else:
            cutoff_freq = None

        if (verbose == Verbosity.BRIEF):
            print("MTF [done]")

        elif (verbose == Verbosity.DETAIL):
            fig = plt.gcf()
            try:
                fig.canvas.manager.set_window_title("MTF")
            except Exception:
                pass
            (ax1,) = plt.subplots(1)
            ax1.plot(interpDistances, interpValues)
            plt.show(block=False)

        return cMTF(interpDistances, interpValues, valueAtNyquist, -1.0), cutoff_freq

    @staticmethod
    def MTF_Full(imgArr, fraction, verbose=Verbosity.NONE):
        """Full pipeline: ESF → LSF → MTF with plots if DETAIL."""
        raw_for_angle = imgArr.copy()
        orig_angle = _raw_edge_angle_deg(raw_for_angle)
        edge_profile_label = "Vertical" if abs(orig_angle) < 45.0 else "Horizontal"

        # Ensure canonical orientation for analysis
        imgArr, _vertical = Transform.Orientify(imgArr)  # FIX: unpack orientify
        esf = MTF.GetESF_crop(imgArr, Verbosity.DETAIL if verbose==Verbosity.DETAIL else Verbosity.NONE)
        lsf = MTF.GetLSF(esf.interpESF, True, Verbosity.DETAIL if verbose==Verbosity.DETAIL else Verbosity.NONE)
        mtf, cutoff_freq = MTF.GetMTF(lsf, fraction, Verbosity.DETAIL if verbose==Verbosity.DETAIL else Verbosity.NONE)
        H, W = imgArr.shape[:2]

        if (verbose == Verbosity.DETAIL):
            plt.figure(figsize=(8, 6))  # new figure so it's not reusing gcf()
            x = [0, np.size(imgArr, 1)-1]
            y = np.polyval(esf.edgePoly, x)

            gs = plt.GridSpec(3, 2, width_ratios=[1.1, 1.4])
            ax1 = plt.subplot(gs[0, 0])   # ROI
            ax2 = plt.subplot(gs[1:2,0])   # Edge profile (ESF)
            #ax3 = plt.subplot(gs[2, 0])   # LSF
            ax4 = plt.subplot(gs[:, 1])   # MTF + info panel

            # Subplot: ROI/Image with detected edge overlay
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax1.axis('off')
            ax1.set_title(f"Original Image\nDimensions: {W} by {H}\nOrientation: {edge_profile_label}")

            # Subplot: Edge profile (raw + interpolated)
            ax2.plot(esf.rawESF.x, esf.rawESF.y, label="Raw ESF")
            ax2.plot(esf.interpESF.x, esf.interpESF.y, label="Smoothed ESF")
            top = np.max(esf.rawESF.y) - esf.threshold
            bot = np.min(esf.rawESF.y) + esf.threshold
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [top, top], color='red')
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [bot, bot], color='red')
            # ax2.set_title(f"Edge profile: {edge_profile_label}")
            ax2.set_xlabel("Distance (pixels)")
            ax2.set_ylabel("Edge profile (linear)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower right", fontsize=8)
            ax2.minorticks_on()

            # # Subplot: LSF
            # ax3.plot(lsf.x, lsf.y)
            # ax3.set_xlabel("Distance (pixels)")
            # ax3.set_ylabel("LSF (normalized)")
            # ax3.grid(True, alpha=0.3)
            # ax3.minorticks_on()
            
            # Orientation + contrast + dims info
            contrast = _michelson_contrast01(imgArr)
            
            # Subplot: MTF with Nyquist line + info box
            ax4.plot(mtf.x, mtf.y)
            cutoff_txt = f"{cutoff_freq:0.2f}" if cutoff_freq is not None else "N/A"
            ax4.set_title(f"MTF{int(fraction*100)}: {cutoff_txt}\nMTF at Nyquist: {mtf.mtfAtNyquist:0.2f}%")
            ax4.plot(0.5, mtf.mtfAtNyquist/100.0, 'o', color='red', linestyle='None', label='Nyquist Frequency', ms=3)
            if cutoff_freq is not None:
                ax4.plot(cutoff_freq, fraction, 'o', color='red', linestyle='None', label=f'MTF{int(fraction*100)} Frequency', ms=3)
            ax4.text(0.5, 0.99,  f"Original edge angle: {orig_angle:.2f}°", ha='left', va='top', transform=ax4.transAxes)
            ax4.text(0.5, 0.94,   f"Normalized edge angle: {esf.angle:.2f}°", ha='left', va='top', transform=ax4.transAxes)
            ax4.text(0.5, 0.90, f"Width: {esf.width:0.2f} px", ha='left', va='top', transform=ax4.transAxes)
            ax4.text(0.5, 0.85, f"Threshold: {esf.threshold:0.2f}", ha='left', va='top', transform=ax4.transAxes)
            ax4.text(0.5, 0.80, f"Chart contrast: {contrast*100:.1f}%", ha='left', va='top', transform=ax4.transAxes)
            ax4.set_xlabel('Normalized Frequency')
            ax4.set_ylabel('MTF Value')
            ax4.minorticks_on()

         
            

            plt.tight_layout()
            plt.show()

        return cMTF(mtf.x, mtf.y, mtf.mtfAtNyquist, esf.width)

# -----------------------------
# Angle helper
# -----------------------------

def _raw_edge_angle_deg(arr01):
    import numpy as _np, cv2 as _cv2, math as _math
    a = _np.asarray(arr01, dtype=float)
    if a.max() > 1.0:  # normalize if not already 0..1
        a = a / 255.0
    a = _np.squeeze(a)
    if a.ndim == 3 and a.shape[2] >= 3:
        # luminance
        a = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
    edgeImg = _cv2.Canny(_np.uint8(_np.clip(a,0,1)*255), 40, 90, L2gradient=True)
    line = _np.argwhere(edgeImg == 255)
    if line.size < 2:
        return 0.0
    edgePoly = _np.polyfit(line[:,1], line[:,0], 1)
    angle = _math.degrees(_math.atan(-edgePoly[0]))
    return float(angle)

# -----------------------------
# Example batch (commented)
# -----------------------------

# import os
# # Main working directory.
# images_folder = "\\\\Your own image folder name"
# dir = "Your own directory"
# os.chdir(dir + images_folder)
# print("Currently working in " + dir + images_folder)
#
# # Image processing for all in folder that ends with .png
# for i in os.listdir():
#     if i.endswith(".png"):
#         print("Processing image: " + i)
#         fraction = 0.5  # your desired MTF. e.g. 0.5 for MTF50
#         filename = i.replace('.png', '_mtf.png')
#         img = Transform.LoadImg(i)
#         imgArr = Transform.Arrayify(img)
#         res = MTF.MTF_Full(imgArr, fraction, verbose=Verbosity.DETAIL)
#         plt.savefig(filename, bbox_inches='tight', dpi=300)
#         plt.close('all')
#
# print("Done.")

