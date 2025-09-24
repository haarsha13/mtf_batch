# -*- coding: utf-8 -*-
""""

This module provides tools to analyze the resolution and sharpness of images using the Modulation Transfer Function (MTF) method. 
It includes functions for loading images, processing edge spread functions (ESF), line spread functions (LSF), and calculating MTF values.


High-level pipeline (per image/patch)
-------------------------------------
1) Normalize and orient the patch so that the bright/dark quadrants are in a
   consistent arrangement (Transform.Orientify).
2) Detect the edge with Canny and fit a line y = m*x + b (MTF.GetESF_crop).
3) Build the *raw* ESF by projecting pixels onto the normal direction of the
   fitted edge and sorting by distance (MTF.GetESF).
4) Smooth and uniformly re-sample the ESF (cubic spline) and **center** it so that
   the 50% intensity point (edge location) is at x = 0 (MTF.Center_ESF).
5) Differentiate the ESF to obtain the LSF and optionally normalize it (MTF.GetLSF).
6) Apply a Kaiser window (β configurable) to the LSF to reduce spectral leakage,
   then FFT to obtain the MTF. Interpolate onto normalized spatial frequency
   ∈ [0, 1], where 0.5 = Nyquist (MTF.GetMTF).
7) Report metrics and (optionally) plot intermediate steps (MTF.MTF_Full).
"""

# -*- coding: utf-8 -*-
"""Resolution and Sharpness of Images

Original file is located at
  https://colab.research.google.com/drive/1Dt-SLffFv1G7dlGy5o08Gmy8g_18ySrE

# Resolution and Sharpness of Images
From Damian Howe and Haarsha Krishna

## Formatting Data and Classes
"""

import matplotlib.pyplot as plt
import pylab as pylab
import numpy as np
import cv2 as cv2
import math as math
import pandas as pd
import json


from PIL import Image, ImageOps
from scipy import interpolate
from scipy.fft import fft
from scipy.signal import windows
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

@dataclass
class cSet:
  x: np.ndarray # storing indexes as a numpy array
  y: np.ndarray # storing values as a numpy array

@dataclass
class cESF:
  rawESF: cSet # raw ESF data as numpy arrays
  interpESF: cSet # interpolated ESF data as numpy arrays
  threshold: float # threshold used for ESF/MTF
  width: float # pixel transition size
  angle: float # slant angle (deg)
  edgePoly: np.ndarray # polynomial of slant

@dataclass
class cMTF:
  x: np.ndarray # array for indexes
  y: np.ndarray # array for values
  mtfAtNyquist: float # MTF value at sampling speed limit (Nyquist Freq)
  width: float # pixel transition size

@dataclass
class Verbosity(Enum): # output types/level of output
  NONE = 0 # none
  BRIEF = 1 # text
  DETAIL = 2 # graphical

@dataclass
class MTFReport:
    filename: str                    # source image filename
    image_w: int                     # image dimension width (pixels)    
    image_h: int                     # image dimension height (pixels)
    edge_profile: str                # "Vertical" or "Horizontal"
    angle_deg: float                 # edge angle in degrees
    width_px: float                  # pixel transition size
    threshold: float                 # threshold used for ESF/MTF
    contrast: float                  # Michelson contrast (0..1 scale)
    mtf_fraction: float              # fraction of MTF to report (e.g. 0.5 for MTF50)
    mtf50_freq: float | None         # frequency (normalized) where MTF crosses `mtf_fraction`
    mtf_at_nyquist: float            # MTF value at 0.5 (0..1 scale)
    nyquist_frequency: float         # always 0.5 (normalized)
    mtf_x: np.ndarray                # full MTF x (normalized frequency)
    mtf_y: np.ndarray                # full MTF y (values)
    lsf_x: np.ndarray                # optional: LSF x
    lsf_y: np.ndarray                # optional: LSF y

"""## Image importing Functions"""

class Transform:

  @staticmethod
  def _raw_edge_angle_deg(arr01):
    a = np.asarray(arr01, dtype=float)
    if a.max() > 1.0:  # normalize if not already 0..1
      a = a / 255.0
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[2] >= 3:
      # luminance
      a = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
    edgeImg = cv2.Canny(np.uint8(np.clip(a,0,1)*255), 40, 90, L2gradient=True)
    line = np.argwhere(edgeImg == 255)
    if line.size < 2:
      return 0.0
    edgePoly = np.polyfit(line[:,1], line[:,0], 1) # y = mx + b, a polynomial for the edge
    angle = math.degrees(math.atan(-edgePoly[0])) # angle of the edge in degrees
    return float(angle) 

  @staticmethod
  def _otsu_threshold01(gray):
    g = np.clip(gray, 0.0, 1.0)
    hist, bin_edges = np.histogram(g.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(float)
    prob = hist / (hist.sum() + 1e-12)
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
    img = Image.open(file)
    SHAPE_x, SHAPE_y = img.size
    if img.mode in {'I;16','I;16L','I;16B','I;16N'}: # need correct format 16-bit unsigned integer pixel
        gsimg = img
    else:
        gsimg = img.convert('L') # converts to 8 bit pixels in greyscale    
    return gsimg, SHAPE_x, SHAPE_y 

  @staticmethod
  def Arrayify(img):
    if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
      arr = np.asarray(img, dtype = np.double)/65535 # normalizing
    else:
      arr = np.asarray(img, dtype = np.double)/255 # also normalizing
    return arr # returns normalized array of img data

  @staticmethod
  def Imagify(Arr):
    img = Image.fromarray(Arr*255, mode='L')
    return img

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
      pass
    elif(cornerIndexes[0] + cornerIndexes[1]) == 5:
      Arr = np.flip(Arr, axis=0)
      vertical = 1
    elif(cornerIndexes[0] + cornerIndexes[1]) == 2:
      Arr = np.transpose(Arr)
      vertical = 0
    elif(cornerIndexes[0] + cornerIndexes[1]) == 4:
      Arr = np.flip(np.transpose(Arr), axis=0)
      vertical = 0

    return Arr, vertical # returns the array and whether it is vertical or not (1 for vertical, 0 for horizontal)

"""## MTF Functions"""

class MTF:
  # distances = microns/pixels away from the slant edge
  # values = array of brightnesses
  # head = top limit
  # tail = bottom limit

  @staticmethod
  def Center_ESF(esf_raw, esf_interp, verbose=Verbosity.NONE):
    """
    Shift ESF x-axes so the 50% intensity point (half-level) is at x = 0.
    Uses the smoothed/interpolated ESF to estimate the edge location,
    then applies the same shift to BOTH raw and interpolated ESF.

    Returns:
        (esf_raw_centered: cSet, esf_interp_centered: cSet, x_edge: float)
    """
    Imin = float(np.min(esf_interp.y))
    Imax = float(np.max(esf_interp.y))
    half = (Imin + Imax) * 0.5

    # Find where smoothed ESF is closest to half-level
    idx = int(np.argmin(np.abs(esf_interp.y - half)))
    x_edge = float(esf_interp.x[idx])

    esf_raw_c   = cSet(esf_raw.x   - x_edge, esf_raw.y)
    esf_interp_c = cSet(esf_interp.x - x_edge, esf_interp.y)

    if verbose == Verbosity.BRIEF:
        print(f"ESF Center [done] (edge @ x={x_edge:0.3f} → shifted to 0)")

    return esf_raw_c, esf_interp_c, x_edge

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

    if isIncrementing == False:
        distances = -distances

    return cSet(distances[h:t], values[h:t])

  @staticmethod
  def GetESF(Arr, edgePoly, verbose = Verbosity.NONE):
    Y = Arr.shape[0]
    X = Arr.shape[1]

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

    if (distances[0] > distances[-1]):
      distances = np.flip(distances)
      values = np.flip(values)

    if (verbose == Verbosity.BRIEF):
      print(f"Raw ESF [done] (Distance from {0:2.2f} to {1:2.2f})".format(sign*distances[0], sign*distances[-1]))

    elif (verbose == Verbosity.DETAIL):
      x = [0, np.size(Arr,1)-1]
      y = np.polyval(edgePoly, x)

      fig = pylab.gcf()
      fig.canvas.manager.set_window_title('Raw ESF')
      (ax1, ax2) = plt.subplots(2)
      ax1.imshow(Arr, cmap = 'gray')
      ax1.plot(x, y, color = 'r')
      ax2.plot(distances, values)
      plt.show()
      plt.show(block = False)

    return cSet(distances, values)

  @staticmethod
  def GetESF_crop(Arr, verbose = Verbosity.NONE):
    imgArr, verticality = Transform.Orientify(Arr)
    edgeImg = cv2.Canny(np.uint8(imgArr*255), 40, 90, L2gradient = True)

    line = np.argwhere(edgeImg == 255)
    edgePoly = np.polyfit(line[:,1], line[:,0], 1)
    angle = math.degrees(math.atan(-edgePoly[0]))

    finalEdgePoly = edgePoly.copy()
    if angle>0:
      imgArr = np.flip(imgArr, axis = 1)
      finalEdgePoly[1] = np.polyval(edgePoly, np.size(imgArr, 1)-1)
      finalEdgePoly[0] = -edgePoly[0]

    esf = MTF.GetESF(imgArr, finalEdgePoly, Verbosity.BRIEF)

    esf_Values = esf.y
    esf_Distances = esf.x

    max = np.amax(esf_Values)
    min = np.amin(esf_Values)

    threshold = (max - min) * 0.1

    head = np.amax(esf_Distances[(np.where(esf_Values < min + threshold))[0]])
    tail = np.amin(esf_Distances[(np.where(esf_Values > max - threshold))[0]])

    width = abs(head-tail)

    esfRaw = MTF.crop(esf_Values, esf_Distances, head - 1.2*width, tail + 1.2*width)

    qs = np.linspace(0,1,20)[1:-1]
    knots = np.quantile(esfRaw.x, qs)
    tck = interpolate.splrep(esfRaw.x, esfRaw.y, t=knots, k=3)
    ysmooth = interpolate.splev(esfRaw.x, tck)

    InterpDistances = np.linspace(esfRaw.x[0], esfRaw.x[-1], 500)
    InterpValues = np.interp(InterpDistances, esfRaw.x, ysmooth)

    esfInterp = cSet(InterpDistances, InterpValues)

    # --- NEW: center both raw & interp ESF at the 50% level ---
    esfRaw_c, esfInterp_c, x_edge = MTF.Center_ESF(esfRaw, esfInterp, verbose=verbose)

    if (verbose == Verbosity.BRIEF):
        print(f"ESF Crop [done] (Distance from {esfRaw_c.x[0]:2.2f} to {esfRaw_c.x[-1]:2.2f})")

    elif (verbose == Verbosity.DETAIL):
        x = [0, np.size(imgArr,1)-1]
        y = np.polyval(finalEdgePoly, x)

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('ESF Crop (centered)')
        (ax1, ax2) = plt.subplots(2)
        ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
        ax1.plot(x, y, color='red')
        ax2.plot(esfRaw_c.x, esfRaw_c.y, label='Raw (centered)')
        ax2.plot(esfInterp_c.x, esfInterp_c.y, label='Smooth (centered)')
        ax2.axvline(0.0, linestyle='--')
        ax2.legend()
        plt.show(block=False)
        plt.show()


    return cESF(esfRaw_c, esfInterp_c, threshold, width, angle, edgePoly)


  @staticmethod
  def Simplify_ESF(ESF, verbose=Verbosity.NONE):

    res = np.unique(ESF.x, return_index=True, return_counts=True)

    indexes = res[1]
    counts = res[2]
    sz = np.size(res[0])

    distances = ESF.x[indexes]
    values = np.zeros(sz, dtype=np.float64)

    for i in range(sz):
        values[i] = np.sum(ESF.y[indexes[i]:indexes[i]+counts[i]])/counts[i]

    if (verbose == Verbosity.BRIEF):
        print("ESF Simplification [done] (Size from {0:d} to {1:d})".format(np.size(ESF.x), np.size(distances)))

    elif (verbose == Verbosity.DETAIL):
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title("ESF Simplification (Size from {0:d} to {1:d})".format(np.size(ESF.x), np.size(distances)))
        (ax1, ax2) = plt.subplots(2)
        ax1.plot(ESF.x, ESF.y)
        ax2.plot(distances, values)
        plt.show(block=False)
        plt.show()

    return cSet(distances, values)

  @staticmethod
  def GetLSF(ESF, normalize = True, verbose = Verbosity.NONE):
    lsfDividend = np.diff(ESF.y)
    lsfDivisor = np.diff(ESF.x)

    lsfValues = np.divide(lsfDividend, lsfDivisor)
    lsfDistances = ESF.x[0:-1]

    if normalize:
      lsfValues = lsfValues / (max(lsfValues))

    if (verbose == Verbosity.BRIEF):
      print(f"MTF [done]")

    elif (verbose == Verbosity.DETAIL):
      fig = pylab.gcf()
      fig.canvas.manager.set_window_title("LSF")
      (ax1) = plt.subplots(1)
      ax1.plot(lsfDistances, lsfValues)
      plt.show(block=False)
      plt.show()
    return cSet(lsfDistances, lsfValues)

  @staticmethod
  def GetMTF(LSF, fraction, verbose = Verbosity.NONE, beta=14):
    N = np.size(LSF.x)
    px = N/(LSF.x[-1]- LSF.x[0])

    window = windows.kaiser(N,beta=beta)
    windowed_y = LSF.y * window

    values = 1/np.sum(windowed_y)*abs(fft(windowed_y))

    #values = 1/np.sum(LSF.y)*abs(fft(LSF.y))
    
    distances = np.arange(0,N)/N*px

    interpDistances = np.linspace(0,1,200)
    interp = interpolate.interp1d(distances, values, kind='cubic')
    interpValues = interp(interpDistances)
    valueAtNyquist = float(interp(0.5)) * 100

    target = fraction

    crossing_idx = np.where(interpValues <= fraction)[0]

    if len(crossing_idx) > 0:
        i = crossing_idx[0]
        # Linear interpolation for better accuracy
        x0, y0 = interpDistances[i-1], interpValues[i-1]
        x1, y1 = interpDistances[i], interpValues[i]
        cutoff_freq = x0 + (fraction - y0) * (x1 - x0) / (y1 - y0)
    else:
        cutoff_freq = None


    if (verbose == Verbosity.BRIEF):
      print(f"MTF [done]")
    elif (verbose == Verbosity.DETAIL):
      fig = pylab.gcf()
      fig.canvas.manager.set_window_title(f"MTF ({0:2.2f}% at Nyquist)".format(valueAtNyquist))
      (ax1) = plt.subplots(1)
      ax1.plot(interpDistances, interpValues)
      plt.show()
      plt.show(block=False)

    return cMTF(interpDistances, interpValues, valueAtNyquist, -1.0), cutoff_freq, windowed_y, window

  @staticmethod
  def MTF_Full(imgArr_orig, fraction, w=None, h=None, verbose=Verbosity.NONE):
    
    contrast = Transform._michelson_contrast01(imgArr_orig)
    imgArr, verticality = Transform.Orientify(imgArr_orig)
    w, h = imgArr.shape[0], imgArr.shape[1]
    esf = MTF.GetESF_crop(imgArr, Verbosity.DETAIL)  # so you see raw ESF plot
    lsf = MTF.GetLSF(esf.interpESF, True, Verbosity.DETAIL)  # see LSF plot
    mtf, cutoff_freq, windowed_y , window = MTF.GetMTF(lsf, fraction, Verbosity.DETAIL, beta=14)  # see MTF plot

    if verticality > 0:
        verticality = "Vertical"
    else:
        verticality = "Horizontal"

    if (verbose == Verbosity.DETAIL):

        MTF_lens = []
        FREQUENCY = []
        for i in range(len(mtf.y)):
          
          x = mtf.x[i]
          y = mtf.y[i]

          f_nyq = 1 #Normalized Nyquist Frequency
          freqs = x/f_nyq
          MTF_detec = float(np.abs(np.sinc(freqs)))

          if MTF_detec < 1e-3:
            MTF_detec = 1e-3 # to avoid division by zero
          
          MTF_sys = float(y)
          lens = MTF_sys/MTF_detec
          lens = np.clip(lens, 0, 1)
          
          MTF_lens.append(lens)
          FREQUENCY.append(freqs)

        FREQUENCY = np.array(FREQUENCY) 
        MTF_lens = np.array(MTF_lens)

        fig1 = plt.figure(figsize=(6,4)) # new figure so it's not reusing gcf()
        
        plt.plot(FREQUENCY, MTF_lens, label='Lens MTF', color='green')
        plt.xlabel("Spatial Frequency (cyc/pixel)")
        plt.ylabel("Theoretical Lens MTF")
        plt.title("Theoretical Lens MTF Curve")
        plt.grid(True, alpha=0.4)
        plt.close(fig1)


        fig2 = plt.figure(figsize=(8,6)) # new figure so it's not reusing gcf()

        x = [0, np.size(imgArr,1)-1]
        y = np.polyval(esf.edgePoly, x)

        gs = plt.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[2, 0])
        ax4 = plt.subplot(gs[:, 1])

        ax1.imshow(imgArr_orig, cmap='gray', vmin=0.0, vmax=1.0)
        ax1.plot(x, y, color='red', label = 'Edge After Orientation')
        ax1.axis('off')
        ax1.set_title(f"Image Dimensions: {w} by {h}\n Edge Profile: {verticality}")
        ax1.legend(loc='lower left', fontsize=5)
        ax2.plot(esf.rawESF.x, esf.rawESF.y,
                 esf.interpESF.x, esf.interpESF.y)
        top = np.max(esf.rawESF.y)-esf.threshold
        bot = np.min(esf.rawESF.y)+esf.threshold
        ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [top, top], color='red')
        ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [bot, bot], color='red')
        ax2.xaxis.set_visible(True)
        ax2.yaxis.set_visible(True)
        ax2.grid(True)
        ax2.minorticks_on()

        ax3.plot(lsf.x, windowed_y, label='LSF', alpha = 0.7)
        ax3.xaxis.set_visible(True)
        ax3.yaxis.set_visible(True)
        ax3.grid(True)
        ax3.minorticks_on()
        ax3.legend(loc='upper right', fontsize=5)

        ax4.plot(mtf.x, mtf.y)
        ax4.set_title(f"MTF{int(fraction*100)}: {cutoff_freq:0.3f}\nMTF at Nyquist: {mtf.mtfAtNyquist/100:0.5f}")
        ax4.plot(0.5, mtf.mtfAtNyquist/100, 'o', color='red', linestyle='None', label='Nyquist Frequency', ms=3)
        ax4.plot(cutoff_freq, fraction, 'o', color='red', linestyle='None', label=f'MTF{fraction*100} Frequency', ms=3)
        ax4.text(0.5, 0.99, f"Angle: {esf.angle:0.3f} degrees", ha='left', va='top')
        ax4.text(0.5, 0.94, f"Width: {esf.width:0.3f} pixels", ha='left', va='top')
        ax4.text(0.5, 0.89, f"Threshold: {esf.threshold:0.3f}", ha='left', va='top')
        ax4.text(0.5, 0.84, f"Contrast: {contrast*100:0.1f}%", ha='left', va='top')
        ax4.set_xlabel('Normalized Frequency')
        ax4.set_ylabel('MTF Value')
        ax4.minorticks_on()

        plt.tight_layout()
        
    if verbose == Verbosity.DETAIL:
       return cMTF(mtf.x, mtf.y, mtf.mtfAtNyquist, esf.width), fig2, fig1
    else:
       return cMTF(mtf.x, mtf.y, mtf.mtfAtNyquist, esf.width), fig2, None

  @staticmethod
  def analyze(imgArr, filename="", fraction=0.5, beta=14):
    """Pure analysis (metrics+arrays), no plotting."""
    contrast = Transform._michelson_contrast01(imgArr)
    imgArr2, verticality = Transform.Orientify(imgArr)
    w, h = imgArr2.shape[0], imgArr2.shape[1]

    esf = MTF.GetESF_crop(imgArr2, Verbosity.NONE)
    lsf = MTF.GetLSF(esf.interpESF, True, Verbosity.NONE)
    mtf, cutoff_freq, _, _ = MTF.GetMTF(lsf, fraction, Verbosity.NONE, beta=beta)

    edge_profile = "Vertical" if verticality > 0 else "Horizontal"
    mtf_at_nyquist = float(mtf.mtfAtNyquist) / 100.0  # your code reports %; convert to 0..1

    return MTFReport(
            filename=str(filename),
            image_w=int(w), image_h=int(h),
            edge_profile=edge_profile,
            angle_deg=float(esf.angle),
            width_px=float(esf.width),
            threshold=float(esf.threshold),
            contrast=float(contrast),
            mtf_fraction=float(fraction),
            mtf50_freq=(float(cutoff_freq) if cutoff_freq is not None else None),
            mtf_at_nyquist=mtf_at_nyquist,
            nyquist_frequency=0.5,
            mtf_x=mtf.x, mtf_y=mtf.y,
            lsf_x=lsf.x, lsf_y=lsf.y
    )

    # Back-compat alias if you already called AnalyzeToReport elsewhere
    AnalyzeToReport = analyze

  @staticmethod
  def run(imgArr, fraction=0.5, plot=False, verbose=Verbosity.NONE, filename="", beta=14):
        """
        Single public entrypoint:
          - always does analysis and returns an MTFReport
          - if plot=True, also renders the usual figure and returns the Matplotlib Figure
        """
        rep = MTF.analyze(imgArr, filename=filename, fraction=fraction, beta=beta)

        fig2 = None
        fig1 = None

        if plot:
            # reuse your existing full-figure function
            _data_, fig2, fig1 = MTF.MTF_Full(imgArr, fraction, verbose=verbose)
            plt.close(fig2)  # avoid showing it now; caller can show if they want
    
        return rep, fig2, fig1
