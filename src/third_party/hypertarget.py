from __future__ import print_function

from csqa import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal
from scipy.spatial import ConvexHull, Delaunay
from skimage.feature import peak_local_max

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

def window1D(npt=500, filter_frac=0.25):
    """
    Create a 1D window function with a flat central region.
    Slightly more general than scipy.signal.tukey (which is a special case with filter_frac=0.5)

    @param npt:                         length of 1D array to generate
    @param filter_frac:                 fraction of array on each side in smooth cosine function (0<=f<=0.5)

    @return:                            1D window array
    """
    if (filter_frac < 0) or (filter_frac > 0.5):
        _log.critical(f"Filter parameter out of range {filter_frac}")
        filter_frac = max(0, min(0.5, filter_frac))

    fw = filter_frac * npt
    ifw = int(fw)
    filter = np.ones(npt)

    filter[0:ifw] = (1.0 - np.cos(np.pi * np.arange(ifw) / fw)) / 2.0
    filter[-1 : (-ifw - 1) : -1] = (1.0 - np.cos(np.pi * np.arange(ifw) / fw)) / 2.0
    return filter


def window2D(npt=500, filter_frac=0.25):
    """
    Create a 2D window function with a flat central region.

    @param npt:                         dimensions of 2D array to generate
    @param filter_frac:                 fraction of array on each side in smooth cosine function (0<=f<=0.5)

    @return:                            2D window array
    """
    if np.array([npt]).size == 1:
        npt = np.array([npt, npt]).reshape([2])

    filter1D_y = window1D(npt[0], filter_frac)
    filter1D_x = window1D(npt[1], filter_frac)
    return np.outer(filter1D_y, filter1D_x)


def window2D_sym(npt=[500, 500], filter_frac=0.25):
    """
    @param npt:                         dimensions of 2D array to generate
    @param filter_frac:                 fraction of array on each side in smooth cosine function (0<=f<=0.5)

    @return:                            2D window array
    """
    min_npt = np.min(npt)
    central_window = window2D(min_npt, filter_frac=filter_frac)
    if np.array([npt]).size == 1 or npt[0] == npt[1]:
        return central_window

    window = np.zeros(npt)
    window[
        (npt[0] - min_npt) // 2 : (npt[0] - min_npt) // 2 + min_npt,
        (npt[1] - min_npt) // 2 : (npt[1] - min_npt) // 2 + min_npt,
    ] = central_window
    return window


def window_image(im, filter_frac=0.25, remove_mean=False, shift=False):
    """
    Generate a windowed image optionally with zero mean and shifted origin

    @param im:                          original image
    @param filter_frac:                 filter fraction for window function
    @param remove_mean:                 True to remove the mean (at least approximately)
    @param shift:                       True if window is applied relative to corner rather than centre of image

    @return:                            windowed image
    """
    # assert im.shape[0] == im.shape[1], "Code currently assumes square image"
    if len(im.shape) != 2:
        _log.critical("Non grayscale image passed to window function")

    im2 = im.copy()
    filter = window2D(im.shape, filter_frac=filter_frac)
    if shift:
        im2 = np.fft.fftshift(im2)

    if remove_mean:
        im2 = filter * (im2 - np.mean(im2))  # could repeat this step to make mean closer to zero
    else:
        im2 = filter * im2

    if shift:
        im2 = np.fft.ifftshift(im2)

    return im2


def load_image(
    filename,
    npt=None,
    filter_frac=0.25,
    downsample=1,
    offset_yx=None,
    return_unfiltered=False,
    make_images_symmetric=False,
):
    """
    Load an image, optionally crop and window with zero mean

    @param filename:                    location of image file
    @param crop:                        None for no crop, otherwise a 4 element array/tuple/list of left, top, right,
                                        bottom coordinates of crop
    @param filter_frac:                 filter fraction for window function

    @return:                            windowed image (square)
    """
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if downsample != 1:
        im = cv2.resize(im, (im.shape[1] // downsample, im.shape[0] // downsample), interpolation=cv2.INTER_CUBIC)

    if offset_yx is not None:
        rows, cols = im.shape
        M = np.float32([[1, 0, offset_yx[1]], [0, 1, offset_yx[0]]])
        im = cv2.warpAffine(im, M, (cols, rows))

    if npt != None and npt != 0:
        if np.array(npt).size == 1:
            npt = [npt, npt]

        for i in range(2):
            if npt[i] is None:
                npt[i] = im.shape[i]

        npt_ds = np.int32(npt) // downsample
        npt_im_ds = np.int32(im.shape)
        if not np.all(npt_ds <= npt_im_ds):
            _log.critical("Image too small to crop")
        else:
            crop = list((npt_im_ds - npt_ds) // 2) + list((npt_im_ds + npt_ds) // 2)
            crop = np.int32(np.maximum([0, 0, 0, 0], np.minimum(im.shape + im.shape, crop)))
            im = im[crop[0] : crop[2], crop[1] : crop[3]]

    elif make_images_symmetric:
        im = im[0 : np.min(im.shape), 0 : np.min(im.shape)]

    if filter_frac > 0:
        filt_im = window_image(im, filter_frac=filter_frac, remove_mean=True)

    if return_unfiltered:
        return filt_im, im
    else:
        return filt_im


def pad_image(im, pad_factor=2, filter_frac=0.25, window=True, shift=True):
    """
    Pad an image, possibly with window applied first

    @param im:                          original image
    @param pad_factor:
    @param filter_frac:                 filter fraction for window function
    @param window:                      True to apply a window function
    @param shift:                       True if padding is relative to corner rather than centre of image

    @return:                            padded image
    """
    npts = im.shape[0]
    npts_pad = pad_factor * npts
    im_pad = np.zeros((npts_pad, npts_pad), dtype=im.dtype)
    im2 = im.copy()

    if shift:
        im2 = np.fft.fftshift(im2)

    if window:
        im2 = window_image(im2, filter_frac=filter_frac)

    im_pad[
        (npts_pad // 2 - npts // 2) : (npts_pad // 2 + npts // 2),
        (npts_pad // 2 - npts // 2) : (npts_pad // 2 + npts // 2),
    ] = im2

    if shift:
        im_pad = np.fft.ifftshift(im_pad)

    return im_pad


def centre_crop(im, npt_crop, shift=False, offset_yx=[0, 0]):
    """
    Generate a crop of an image

    @param im:                          original image
    @param npt_crop:                    size of crop
    @param shift:                       True if crop is relative to corner rather than centre of image
    @param offset_yx:                    offset of cropped image

    @return:                            cropped image
    """
    if npt_crop is None:
        npt_crop = min(npt_crop, np.min(im.shape[0:2]))
    else:
        npt_crop = min(npt_crop, np.min(im.shape[0:2]))

    npt = np.int32(im.shape[0:2])
    start = np.int32(offset_yx) + npt // 2 - npt_crop // 2
    end = start + npt_crop
    if shift:
        return np.fft.fftshift(im)[start[0] : end[0], start[1] : end[1]]
    else:
        return im[start[0] : end[0], start[1] : end[1]]



def field_positions(imshape, ifieldsXY, field_extent=[-1, 1, -1, 1]):
    """
    Find field positions in normalised units corresponding to pixel indices based
    on image shape
    """
    nxy = np.array(imshape[1::-1])
    fieldTL = np.array([field_extent[0], field_extent[2]])
    fieldWH = np.array([field_extent[1] - field_extent[0], field_extent[3] - field_extent[2]])
    fieldsXY = fieldTL + (fieldWH * ifieldsXY) / nxy
    return fieldsXY


def field_indices(imshape, fieldsXY, field_extent=[-1, 1, -1, 1]):
    """
    Find pixel indices corresponding to field positions in normalised units based
    on image shape
    """
    nxy = np.array(imshape[1::-1])
    ifieldsXY = np.int32(nxy * (1 + np.array(fieldsXY)) / 2)
    fieldsXY = field_positions(imshape, ifieldsXY=ifieldsXY, field_extent=field_extent)
    return ifieldsXY, fieldsXY


def field_corners(imshape, fieldsXY, field_extent=[-1, 1, -1, 1], return_cent=False):
    """
    Selects the best four corners from a set of fields

    For a given set of focus data sampled over a sensor, define coordinates of corners
    according to the relative positions given

    @param imshape:         image shape (tuple, y then x)
    @param fieldsXY:        list of field positions to select from
    @param field_extent:    left, right, top and bottom extent of field (typically -1 to 1 in both axes)

    returns: the indices in the image corresponding to the corners, and the fractional locations (these may be slightly different to the inputs due to the grid of the images)
    """

    ifieldsXY, fieldsXY = field_indices(imshape=imshape, fieldsXY=fieldsXY, field_extent=field_extent)
    argTL = np.argmax(np.dot(fieldsXY, [-1, -1]))
    argTR = np.argmax(np.dot(fieldsXY, [1, -1]))
    argBL = np.argmax(np.dot(fieldsXY, [-1, 1]))
    argBR = np.argmax(np.dot(fieldsXY, [1, 1]))
    field_strs = ["TL", "TR", "BL", "BR"]
    args = np.int32([argTL, argTR, argBL, argBR])
    if return_cent:
        field_strs += ["C"]
        argC = np.argmin(
            np.linalg.norm(
                np.array(fieldsXY) - np.array([np.mean(field_extent[0:2]), np.mean(field_extent[2:4])]), axis=1
            )
        )
        args = np.int32(list(args) + [argC])

    return np.int32(ifieldsXY)[args], np.array(fieldsXY)[args], field_strs


def plot_hypertarget_overview(
    im, fieldsXY, dxy, title="", show_corners=False, plot_patches=False, field_extent=[-1, 1, -1, 1]
):
    plt.figure(figsize=[9, 7])
    extent = [field_extent[0], field_extent[1], field_extent[3], field_extent[2]]
    plt.imshow(
        im, cmap="gray", extent=extent, aspect=im.shape[0] / im.shape[1]
    )  # NOTE: extent is L, R, B, T - last two terms are swapped
    plt.scatter(fieldsXY[:, 0], fieldsXY[:, 1], color=[1, 1, 0])
    if show_corners:
        ifieldsXYcorn, fieldsXYcorn, field_strs = field_corners(
            imshape=im.shape, fieldsXY=fieldsXY, field_extent=field_extent, return_cent=True
        )

        for i, field_str in enumerate(field_strs):
            plt.scatter(fieldsXYcorn[i : i + 1, 0], fieldsXYcorn[i : i + 1, 1], label=field_str)
        plt.legend()

    plt.title(title)

    if plot_patches:
        npatch = len(fieldsXY)
        nx = int(np.ceil(np.sqrt(npatch)))
        ny = int(np.ceil(npatch / nx))

        ifieldsXY, fieldsXY2 = field_indices(im.shape, fieldsXY, field_extent=field_extent)

        fig, axes = plt.subplots(ny, nx, figsize=(3 * nx, 3 * ny))
        fig.suptitle("%s PATCHES" % title)
        axes = axes.flatten()
        for iax, ax in enumerate(axes):
            if iax < npatch:
                yx0 = np.int32(ifieldsXY[iax, ::-1]) - dxy // 2
                ax.imshow(im[yx0[0] : yx0[0] + dxy, yx0[1] : yx0[1] + dxy], cmap="gray", interpolation="nearest")
            else:
                ax.remove()


class HyperTarget:
    """
    Class to handle coordinates of relevant parts of a hypertarget
    Finds and stores coordinates for textured patches, slant edge patches,
    and flat patches
    """

    def __init__(
        self,
        im_gray,
        nds: int = 16,
        flens: int = 420,
        fcoll: int = 2000,
        target_square_npix: int = 1000,
        target_pix_size: float = 15e-3,
        sensor_pix_size: float = 3.2e-3,
        field_extent=[-1, 1, -1, 1],
        plot_slant: bool = False,
        plot_text: bool = False,
        plot_flat: bool = False,
        plot_patches: bool = False,
    ):
        """
        Simple method just assumes that after blurring (by downsample) the textured
        regions of the target are grey and square with known shape.

        target_square_npix is the size in pixels in the fabricated (etched) target
        which is hard wired to 1000
        """
        self.field_extent = field_extent
        self.imshape = im_gray.shape

        # Work out the dimensions of the textured grey regions
        image_square_npix_ds = int(target_square_npix * (target_pix_size / sensor_pix_size) * (flens / fcoll) / nds)
        im = cv2.resize(im_gray, dsize=(0, 0), fx=1 / nds, fy=1 / nds, interpolation=cv2.INTER_AREA)

        # Simple colour transform as abs difference from mean of image.
        im2 = np.abs(im - np.mean(im))

        # Make a zero mean transformed image convolve with a simple tile based on
        # expected size of square regions
        # NOTE: THIS MAY FAIL FOR LARGE TILT OF PATTERN
        im2 = im2 - np.mean(im2)
        tile = -np.ones((image_square_npix_ds, image_square_npix_ds))

        conv = scipy.signal.convolve2d(im2, tile)
        # plt.figure()
        # plt.imshow(conv)
        # plt.title("%d" % image_square_npix_ds)

        # Best target coordinates occur at local maxima of convolution
        coordinates = (
            peak_local_max(
                conv,
                min_distance=image_square_npix_ds // 2,
                threshold_abs=np.percentile(conv, 85),
                exclude_border=int(1.25 * image_square_npix_ds),
            )
            - image_square_npix_ds // 2
        )

        # Measure slant edge always - angle of target is good to know
        # Edge coordinates
        tri = Delaunay(coordinates)
        ds = []
        ths_deg = []
        for simp in tri.simplices:
            pts = tri.points[simp, :]
            for i in range(3):
                dpt = pts[i, :] - pts[(i + 1) % 3, :]
                ds += [np.linalg.norm(dpt)]
                ths_deg += [((np.rad2deg(np.arctan2(dpt[1], dpt[0])) + 180) % 90) - 45]

        sel = np.where(ds < np.mean(ds))[0]
        ths_deg = [ths_deg[s] for s in sel]
        ds = [ds[s] for s in sel]

        self.slant_angle_deg = np.mean(ths_deg)
        th = np.deg2rad(self.slant_angle_deg)
        d = np.mean(ds) / np.sqrt(2)

        edge_fieldsXY = []
        for this_th in th + np.pi / 2 * np.arange(4):
            edge_coordsR = coordinates + 0.75 * d * np.array([-np.sin(this_th), np.cos(this_th)])
            ifieldsXY_R = np.array([[coord[1], coord[0]] for coord in edge_coordsR])
            fieldsXY_R = field_positions(imshape=im.shape, ifieldsXY=ifieldsXY_R, field_extent=field_extent)
            edge_fieldsXY += list(fieldsXY_R)

        edge_fieldsXY = np.array(edge_fieldsXY)
        keep_edge_fields = np.where(np.max(np.abs(edge_fieldsXY), axis=1) < 0.8)[0]
        self.slant_fieldsXY = edge_fieldsXY[keep_edge_fields]
        self.slant_ifieldsXY = field_indices(imshape=im.shape, fieldsXY=self.slant_fieldsXY, field_extent=field_extent)[
            0
        ]

        # Textured region coordinates and derived corner locations
        self.text_ifieldsXY = np.array([[coord[1], coord[0]] for coord in coordinates])
        self.text_fieldsXY = field_positions(imshape=im.shape, ifieldsXY=self.text_ifieldsXY, field_extent=field_extent)
        self.text_ifieldsXYcorn, self.text_fieldsXYcorn, self.text_field_strs = field_corners(
            imshape=im.shape, fieldsXY=self.text_fieldsXY, field_extent=field_extent, return_cent=True
        )

        # Flat patch target coordinates occur at local minima of convolution
        flat_coordinates = (
            peak_local_max(
                -conv,
                min_distance=image_square_npix_ds // 2,
                threshold_abs=np.percentile(conv, 85),
                exclude_border=int(1.25 * image_square_npix_ds),
            )
            - image_square_npix_ds // 2
        )
        self.flat_ifieldsXY = np.array([[coord[1], coord[0]] for coord in flat_coordinates])
        self.flat_fieldsXY = field_positions(imshape=im.shape, ifieldsXY=self.flat_ifieldsXY, field_extent=field_extent)
        self.flat_ifieldsXYcorn, self.flat_fieldsXYcorn, self.flat_field_strs = field_corners(
            imshape=im.shape, fieldsXY=self.flat_fieldsXY, field_extent=field_extent, return_cent=True
        )

        self.title = "SLANT EDGES @ %.3f$^o$" % self.slant_angle_deg
        self.image_square_npix = int(target_square_npix * (target_pix_size / sensor_pix_size) * (flens / fcoll))
        if plot_slant:
            dxy = 2 * image_square_npix_ds // 5
            _, fieldsXY = self.slant_ifieldsXY, self.slant_fieldsXY
            plot_hypertarget_overview(
                im,
                fieldsXY,
                dxy=dxy,
                title=self.title,
                show_corners=False,
                plot_patches=plot_patches,
                field_extent=field_extent,
            )

        if plot_text:
            dxy = image_square_npix_ds
            title = "FOCUS FIELDS/CORNER %s" % self.title
            _, fieldsXY = self.text_ifieldsXY, self.text_fieldsXY
            plot_hypertarget_overview(
                im,
                fieldsXY,
                dxy=dxy,
                title=title,
                show_corners=True,
                plot_patches=plot_patches,
                field_extent=field_extent,
            )

        if plot_flat:
            dxy = image_square_npix_ds // 2
            title = "FLAT TEXTURES %s" % self.title
            _, fieldsXY = self.flat_ifieldsXY, self.flat_fieldsXY
            plot_hypertarget_overview(
                im,
                fieldsXY,
                dxy=dxy,
                title=title,
                show_corners=False,
                plot_patches=plot_patches,
                field_extent=field_extent,
            )

        # Map coordinates back to image space?
        self.text_ifieldsXY, self.text_fieldsXY = field_indices(
            self.imshape, fieldsXY=self.text_fieldsXY, field_extent=self.field_extent
        )
        self.text_ifieldsXYcorn, self.text_fieldsXYcorn = field_indices(
            self.imshape, fieldsXY=self.text_fieldsXYcorn, field_extent=self.field_extent
        )
        self.slant_ifieldsXY, self.slant_fieldsXY = field_indices(
            self.imshape, fieldsXY=self.slant_fieldsXY, field_extent=self.field_extent
        )
        self.flat_ifieldsXY, self.flat_fieldsXY = field_indices(
            self.imshape, fieldsXY=self.flat_fieldsXY, field_extent=self.field_extent
        )

    def __repr__(self):
        return "Hypertarget: slant %.1fdeg, ntext %d, nslant %d, nflat %d" % (
            self.slant_angle_deg,
            len(self.text_fieldsXY),
            len(self.slant_fieldsXY),
            len(self.flat_fieldsXY),
        )

    def getFieldIndices(self, fieldsXY, im_gray=None):
        if im_gray is None:
            imshape = self.imshape
        else:
            imshape = im_gray.shape

        return field_indices(
            imshape,
            fieldsXY,
        )

    def getPatch(self, im_gray, ifieldXY, patch_size=10):
        yx0 = np.int32(ifieldXY[::-1]) - patch_size // 2
        patch = im_gray[yx0[0] : yx0[0] + patch_size, yx0[1] : yx0[1] + patch_size]

        return patch

    def getDarkNoiseEstimate(self, im_gray):
        varsum = 0
        for ifieldXY in self.flat_ifieldsXY:
            patch = self.getPatch(im_gray, ifieldXY, patch_size=(2 * self.image_square_npix) // 5)
            varsum += np.var(patch)

        return varsum / len(self.flat_ifieldsXY)

    def getDarkBrightNoiseEstimates(self, im_gray):
        darkvar = 0
        brightvar = 0
        darkmean = 0
        brightmean = 0

        nfields = len(self.flat_ifieldsXY)
        xys = np.abs(np.linspace(-1, 1, self.image_square_npix))
        xs, ys = np.meshgrid(xys, xys)
        offset = np.maximum(xs, ys)
        darkpix = np.where(offset < 0.4)
        brightpix = np.where(np.logical_and(offset < 0.9, offset > 0.6))

        for ifieldXY in self.flat_ifieldsXY:
            patch = self.getPatch(im_gray, ifieldXY, patch_size=self.image_square_npix)
            darkvals = patch[darkpix]
            darkvar += np.var(darkvals)
            darkmean += np.mean(darkvals)
            brightvals = patch[brightpix]
            brightvar += np.var(brightvals)
            brightmean += np.mean(brightvals)

        # darkim = np.zeros((self.image_square_npix, self.image_square_npix))
        # darkim[darkpix] = 1
        # brightim = np.zeros((self.image_square_npix, self.image_square_npix))
        # brightim[brightpix] = 1
        # fig, axes = plt.subplots(1, 2, figsize=(10,5))
        # axes[0].imshow(brightim)
        # axes[1].imshow(darkim)

        return darkvar / nfields, darkmean / nfields, brightvar / nfields, brightmean / nfields