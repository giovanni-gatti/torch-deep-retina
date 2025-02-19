"""
Generate commonly used visual stimuli

Functions in this module either generate a numpy array
that encodes a particular stimulus (e.g. the `flash` function
generates a full-field flash, or the `contrast_steps` function
generates a sequence of contrast step changes), or they are used
for composing multiple stimulus sequences (concatenating them)
and converting them into a spatiotemporal stimulus (using
rolling_window) that can be fed to a Keras model (for example).
"""

from __future__ import absolute_import, division, print_function

from itertools import repeat
from numbers import Number

import numpy as np
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean
import skimage.draw
import cv2
import torch
import torch.nn.functional as F
import pyret.filtertools as ft

__all__ = ['concat', 'white', 'contrast_steps', 'flash','spatialize',
                      'bar', 'driftingbar', 'cmask','paired_flashes',
                      'rolling_window','get_cutout',"spatial_pad"]

def get_rf_centers(rfs):
    """
    Finds the row,col coordinate of the center of a receptive field.
    First rank 1 decomposes the receptive field, then uses spatial
    component to find the row,col coordinate with the maximum value.

    rfs: ndarray or list of ndarrays (N,C,H,W)
        a list of spatiotemporal receptive fields.
    """
    centers = []
    for rf in rfs:
        spat, _ = ft.decompose(rf)
        idx = np.argmax(np.abs(spat.ravel()))
        center = np.unravel_index(idx,spat.shape)
        centers.append(center)
    return centers

def get_cutout(stimulus, center, span=20, pad_to=50):
    """ 
    Cutout a spatial subset of the stimulus centered at the
    argued center.

    stimulus: ndarray or torch tensor (T,H,W) or (T,D,H,W)
        the superset of the stimulus of interest
    center: int or list-like (chan,row,col)
        the coordinate for the cutout to be centered on
    span: int
        the height and width of the cutout
    pad_to: int
        pads the cutout with zeros on every side by pad amount
        if 0, no padding occurs
    """
    if pad_to is None or pad_to < span: pad_to = None
    row = (max(0,center[0]-span//2), min(center[0]+span//2+(span%2),
                                                stimulus.shape[-2]))
    col = (max(0,center[1]-span//2), min(center[1]+span//2+(span%2),
                                                stimulus.shape[-1]))
    s = stimulus[...,row[0]:row[1],col[0]:col[1]]
    if pad_to is not None and pad_to > 0:
        s = spatial_pad(s,pad_to)
    return s

def extend_to_edge(img_shape, pt1, pt2):
    """
    Finds the points that cross at the x=0 and x=img_shape[0] edges
    along the line defined by the shortest distance between the two
    argued points. (Assists in drawing a line across the whole image
    using cv2)

    img_shape: tuple of ints with len 2 (height,width)
    pt1: tuple of ints with len 2 (row,col)
    pt2: tuple of ints with len 2 (row,col)
    """
    m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    b = pt1[1]-m*pt1[0]
    new_pt1 = (0,int(b))
    y = m*img_shape[1]+b
    new_pt2 = (img_shape[1], int(y))
    return new_pt1, new_pt2

def center_line_coords(img_shape, angle=0):
    """
    Finds two points that define a line across the center of the image
    with a slope defined by the angle from the rightward horizontal.
    (These points can be used with the cv2.line fxn to draw a line
    with the desired slope crossing through the center of the image)

    img_shape: tuple of ints with len 2 (height,width)
    angle: float or int
        defines the slope of the virtual line in degrees
    """
    img_center = np.array([img_shape[0]//2, img_shape[1]//2])
    # Horizontal
    if angle % 180 == 0:
        coords = [(0, img_center[0]), (img_shape[1], img_center[0])]
    # Vertical
    elif angle % 180 == 90:
        coords = [(img_center[1], 0), (img_center[1], img_shape[0])]
    else:
        rads = (angle-90)*np.pi/180
        vec = np.array([np.sin(rads), np.cos(rads)])
        pt1, pt2 = extend_to_edge(img_shape, np.flip(img_center),
                                         np.flip(img_center)+vec)
        coords = [pt1, pt2]
    return coords

def line_thru_center(img, bar_width, angle=0):
    """
    Draws a line through the center of the image with a slope defined
    by the angle and a line width defined by the barsize.

    img: numpy ndarray with shape (H,W)
        the image to have a line drawn on it
    bar_width: int or float
        defines the width of the line
    angle: int or float
        defines the slope of the line in degrees
    """
    coords = center_line_coords(img.shape, angle)
    cv2.line(img, coords[0], coords[1], 1, bar_width)
    return img

def paint_stripes(img, bar_width=10, angle=0, stripe_val=1):
    """
    Paints stripes onto the argued image, angled from the rightward
    horizontal. Differs from `stripes` in that it paints on a
    preexisting image.

    Inputs:
        img: ndarray (H,W)
            the image to be painted
        bar_width: int greater than 3
            the size of the stripes in terms of pixels
        angle: float
            the stripes are drawn according to the specified
            angle on the unit circle
        stripe_val: float
            the value of the stripe to be painted on.
    """
    img_shape = img.shape
    striped_img = get_striped_img(img.shape, bar_width=bar_width,
                                                     angle=angle)
    img[striped_img==1] = striped_img[striped_img==1]*stripe_val
    return img

def get_striped_img(img_shape, bar_width, angle=0):
    """
    Creates a fresh image of white, black alternating stripes angled
    from the rightward horizontal. Differs from `paint_stripes` in
    that it creates a new image from scratch.

    Inputs:
        img_shape: sequence of ints with length 2 (height,width)
            the size of the image to be returned
        bar_width: int
            the size of the stripes in terms of pixels
        angle: float
            the stripes are drawn according to the specified
            angle on the unit circle
    """
    angle = angle % 360
    if angle % 180 == 0:
        img = np.zeros((bar_width*2, img_shape[1]))
        img[:bar_width] = 1
        img = np.tile(img, (img_shape[0]//(bar_width*2), 1))
        if img.shape[0] < img_shape[0]:
            tup = (img_shape[0]-img.shape[0], img_shape[1])
            img = np.concatenate([img, np.ones(tup)])
        return img
    elif angle % 90 == 0 and angle % 180 == 90:
        img = np.zeros((bar_width*2, img_shape[0]))
        img[:bar_width] = 1
        img = np.tile(img, (img_shape[1]//(bar_width*2), 1))
        if img.shape[0] < img_shape[1]:
            tup = (img_shape[1]-img.shape[0], img_shape[0])
            img = np.concatenate([img, np.ones(tup)])
        return img.T
    assert img_shape[0] >= 2*bar_width and img_shape[1] >= 2*bar_width
    assert bar_width >= 4, "Sorry, just easier to code"
    coords = center_line_coords(img_shape, angle)
    temp_angle = 90-(angle%180)
    y_step = round(bar_width*np.sin(temp_angle*np.pi/180))
    x_step = round(bar_width*np.cos(temp_angle*np.pi/180))
    n_steps_half_x = abs((img_shape[1]//x_step)//2)
    n_steps_half_y = abs((img_shape[0]//y_step)//2)
    n_steps_half = max(n_steps_half_x, n_steps_half_y)
    start_xs = [-x_step*n_steps_half+c[0] for c in coords]
    start_ys = [-y_step*n_steps_half+c[1] for c in coords]
    img = np.zeros(img_shape)
    for i in range(0,int(2*n_steps_half)+1,2):
        new_xs = [int(x+i*x_step) for x in start_xs]
        new_ys = [int(y+i*y_step) for y in start_ys]
        pt1, pt2 = extend_to_edge(img_shape, (new_xs[0], new_ys[0]),
                                             (new_xs[1], new_ys[1]))
        cv2.line(img, pt1, pt2, 1, bar_width)
    return img

def dir_select_vid(img_shape, bar_width, angle, step_size=1,
                                n_frames=90, grating=False):
    """
    Creates a moving bar stimulus in which the moving bar is contained
    within a circle within the image. The bar is angled from the 
    rightward horizontal. The diameter of the circle is the size of
    the image.

    Inputs:
        img_shape: sequence of ints with length 2 (height,width)
            the size of the image to be returned
        bar_width: int
            the size of the stripes in terms of pixels
        angle: float
            the stripes are drawn according to the specified
            angle on a unit circle (counter clockwise from the
            rightward horizontal)
        step_size: int
            the number of pixels to skip until drawing the line in the
            next frame.
        n_frames: int
            the total number of frames in the video.
        grating: bool
            if true, a moving grating is created instead of a single
            moving line.
    Returns:
        video: ndarray (N, H, W)
            the video as a numpy array.
    """
    angle = angle % 360
    double_shape = tuple(i*2 for i in img_shape)
    big_img = np.zeros(double_shape)
    if grating:
        big_img = paint_stripes(big_img, bar_width, angle)
    else:
        big_img = line_thru_center(big_img, bar_width, angle)
    rads = angle*np.pi/180
    ty = float(np.cos(rads))*step_size
    tx = float(np.sin(rads))*step_size
    if angle <= 90 and angle >= 0:
        idxs = np.s_[img_shape[0]:, img_shape[1]:]
    elif angle > 90 and angle < 180:
        idxs = np.s_[:img_shape[0], img_shape[1]:]
    elif angle >= 180 and angle <= 270:
        idxs = np.s_[:img_shape[0], :img_shape[1]]
    else:
        idxs = np.s_[img_shape[0]:, :img_shape[1]]
    rows, cols = double_shape
    M = np.float32([[1,0,0], [0,1,0]])
    frames = []
    center = [s//2 for s in img_shape]
    for i in range(n_frames):
        M[0,-1] = tx*i
        M[1,-1] = ty*i
        new_img = cv2.warpAffine(big_img, M, (cols, rows))
        img = new_img[idxs]
        mask = circle_mask(center, img_shape[0]//2, img_shape)
        img[mask!=1] = 0 # cut out circle for consistent luminance
        frames.append(img)
    return np.asarray(frames)

def motion_reversal(img_shape, start_pt=(0,0), horz_vel=0.5,
                            vert_vel=0, angle=90, bar_width=4,
                            n_frames=None, rev_pt=None):
    """
    Creates a motion reversal stimulus. A bar moves across the screen
    with the corresponding horizontal and vertical velocities. The bar
    then reverses at the specified reversal point. Defaults to the
    center of the image.

    Inputs:
        img_shape: sequence of ints with length 2 (height,width)
            the size of the image to be returned
        start_pt: tuple of ints (row,col)
            the starting point of the line
        horz_vel: float
            the horizontal velocity in pixels/frame
        vert_vel: float
            the vertical velocity in pixels/frame
        angle: float
            the stripes are drawn according to the specified
            angle on a unit circle (counter clockwise from the
            rightward horizontal)
        bar_width: int
            the size of the stripes in terms of pixels
        n_frames: int
            the total number of frames in the video. If None, defaults
            to the number of frames required for the bar to move to
            the desired reversal point and return to its starting pt.
        rev_pt: tuple of ints
            the coordinates of the reversal point. Defaults to the 
            center of the image if None is argued.
    Returns:
        video: ndarray (N, H, W)
            the video as a numpy array.
    """

    if rev_pt is None:
        rev_pt = (img_shape[0]//2, img_shape[1]//2)
    img = np.zeros(img_shape)
    pt1, pt2 = center_line_coords(img_shape, angle)
    cv2.line(img, pt1, pt2, 1, bar_width)

    top_pt = pt1 if pt1[1] < pt2[1] else pt2
    if horz_vel != 0:
        n_shifts = top_pt[0]-start_pt[0]
        start_img = np.roll(img, n_shifts, axis=1)
    if vert_vel != 0:
        n_shifts = top_pt[1]-start_pt[1]
        start_img = np.roll(start_img, n_shifts, axis=0)

    if n_frames is None:
        n_frames = abs(int((rev_pt[0]-start_pt[0])/horz_vel*2))

    frames = []
    cumu_horz = 0
    cumu_vert = 0
    horz_flip = False
    vert_flip = False
    for i in range(n_frames):
        cumu_horz += horz_vel
        new_img = np.roll(start_img, int(cumu_horz), axis=1)
        cumu_vert += vert_vel
        new_img = np.roll(new_img, int(cumu_vert), axis=0)
        new_pt = ((start_pt[0]+cumu_horz)%img_shape[0],
                    (start_pt[1]+cumu_vert)%img_shape[1])
        if horz_vel*new_pt[0] >= horz_vel*rev_pt[0] and not horz_flip:
            horz_vel = -horz_vel
            horz_flip = True
        if vert_vel*new_pt[1] >= vert_vel*rev_pt[1] and not vert_flip:
            vert_vel = -vert_vel
            vert_flip = True
        frames.append(new_img)
    return np.asarray(frames)

def dr_motion_reversal(xflip, speed=0.24):
    c1, _, X1 = driftingbar(velocity=speed, width=2, x=(-40, xflip))
    c2, _, X2 = driftingbar(velocity=speed, width=2, x=(xflip, -40))
    centers = np.concatenate((c1, c2))[40:]
    tflip = np.where(np.diff(centers) == 0)[0][0]
    stim = np.concatenate([X1, X2],axis=0)
    # resp = model.predict(stim)
    return centers, tflip, stim

def reversing_grating(nsamples, halfperiod, barsize, phase, waveform):
    return reverse(grating(barsize=(barsize, 0), phase=(phase, 0),
                                                waveform=waveform),
                                                halfperiod, nsamples)

def circle_mask(center, radius, mask_shape=(50,50)):
    """
    Creates a binary mask in the shape of a circle. 1s are inside the
    circle, 0s are outside.

    center: sequence of ints with len 2
        the coordinates of the center of the circle
    radius: float
        the radius of the circle
    mask_shape: sequence of ints len 2
        the shape of the mask
    """
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                      mask_shape)
    mask = np.zeros(mask_shape)
    mask[row_coords, col_coords] = 1
    return mask
        
def paste_circle(foreground, background, radius, center):
    """
    Pastes the foreground pattern onto the background pattern in a
    window the shape of a circle.

    foreground: ndarray with shape (H, W)
        the pattern that the circle will take
    backgroun: ndarray with shape (H,W)
        the pattern of the background
    radius: float
        the radius of the circle in pixels
    center: sequence of ints with length 2
        the center coordinates of the circle (0,0 is the
        upperleftmost pixel)
    """
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                background.shape)
    img = background.copy()
    img[row_coords, col_coords] = foreground[row_coords, col_coords]
    return img

def periodic_differential_circle(n_frames=100, period_dur=10,
                                        bar_width=4,
                                        inner_bar_width=None,
                                        sync_periods=False,
                                        image_shape=(50,50),
                                        center=(25,25),
                                        n_steps=1,
                                        radius=5,
                                        horizontal_foreground=True,
                                        horizontal_background=True,
                                        background_grating=None,
                                        circle_grating=None):
    """
    Creates circle window onto a grating with a background grating.
    The foreground and background gratings move periodically up and
    down.

    Inputs:
        n_frames: int
            the total number of frames in the movie
        period_dur: int
            the number of frames between each shift
        bar_width: int
            the size of the grating bars in pixels
        inner_bar_width: int
            size of grating bars inside circle. If None, set to
            bar_width
        sync_periods: bool
            True: movements between background and foreground are
            synchronized 
            False: movements between background and foreground are
            asynchronous
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is
            the upper leftmost part of the image.
        n_steps: int
            number of steps taken when shifting
        radius: float
            the radius of the forground circle
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical
        background_grating: ndarray (H,W)
            optionally pass custom background image through
        circle_grating: ndarray (H,W)
            optionally pass custom foreground image through

    Returns:
        frame sequence of shape (n_frames, image_shape[0],
        image_shape[1])
    """
    bar_width=int(bar_width)
    if inner_bar_width is None: 
        inner_bar_width = bar_width
    else:
        inner_bar_width = int(inner_bar_width)
    if background_grating is None:
        angle = 0 if horizontal_background else 90
        background_grating = get_striped_img(image_shape, bar_width,
                                                        angle=angle)
    if circle_grating is None:
        angle = 0 if horizontal_foreground else 90
        circle_grating = get_striped_img(image_shape, inner_bar_width,
                                                         angle=angle)
    background_axis = 0 if horizontal_background else 1
    foreground_axis = 0 if horizontal_foreground else 1
    background_phase = 1
    phase_offset = period_dur//2
    if sync_periods:
        foreground_phase = background_phase 
    else:
        foreground_phase = 1-background_phase
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                            circle_grating.shape)

    frames = []
    shift_dir = -1
    for i in range(n_frames):
        imoddur = i % period_dur
        if imoddur >= 0 and imoddur < n_steps:
            shift_dir = -shift_dir if imoddur == 0 else shift_dir
            if background_phase == 0:
                background_grating = np.roll(background_grating,
                                               shift_dir,
                                               axis=background_axis)
            if foreground_phase == 0:
                circle_grating = np.roll(circle_grating, shift_dir,
                                              axis=foreground_axis)
        elif imoddur>=phase_offset and imoddur<phase_offset+n_steps:
            if background_phase == 1:
                background_grating = np.roll(background_grating,
                                            shift_dir,
                                            axis=background_axis)
            if foreground_phase == 1:
                circle_grating = np.roll(circle_grating, shift_dir,
                                                axis=foreground_axis)
                
        new_frame = background_grating.copy()
        new_frame[row_coords, col_coords] = circle_grating[row_coords,
                                                           col_coords]
        frames.append(new_frame)
    return np.asarray(frames), background_grating, circle_grating

def random_differential_circle(n_frames=100, bar_width=4,
                                    inner_bar_width=None,
                                    foreground_velocity=1,
                                    sync_jitters=False,
                                    background_velocity=1,
                                    image_shape=(50,50),
                                    center=(25,25),
                                    radius=5,
                                    horizontal_foreground=True,
                                    horizontal_background=True,
                                    seed=None,
                                    background_grating=None,
                                    circle_grating=None):
    """
    Creates circle window onto a grating with a background grating.
    The foreground and background gratings jitter at different rates.

    Inputs:
        n_frames: int
            the total number of frames in the movie
        bar_width: int
            the size of the grating bars in pixels
        inner_bar_width: int
            size of grating bars inside circle. If None, set to
            bar_width
        sync_jitters: bool
            True: movements between background and foreground are
                  synchronized (uses background velocity for jitters)
            False: movements between background and foreground are
                  asynchronous
        foreground_velocity: float
            the foreground (grating circle) movement intensity.
            units of pixels per frame
        background_velocity: float
            the background (grating) movement intensity.
            units of pixels per frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is
            the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical
        background_grating: ndarray (H,W)
            optionally pass custom background image through
        circle_grating: ndarray (H,W)
            optionally pass custom foreground image through

    Returns:
        frame sequence of shape (n_frames, image_shape[0],
                                            image_shape[1])

    """
    if seed is not None:
        np.random.seed(seed)
    bar_width=int(bar_width)
    if inner_bar_width is None: 
        inner_bar_width = bar_width
    else:
        inner_bar_width = int(inner_bar_width)
    if circle_grating is None:
        angle = 0 if horizontal_foreground else 90
        circle_grating = get_striped_img(image_shape, inner_bar_width,
                                                         angle=angle)
    if background_grating is None:
        angle = 0 if horizontal_background else 90
        background_grating = get_striped_img(image_shape, bar_width,
                                                        angle=angle)
    background_axis = 0 if horizontal_background else 1
    foreground_axis = 0 if horizontal_foreground else 1
    background_steps = np.round(np.random.random(n_frames))
    background_steps = background_steps.astype(np.int)
    background_steps[background_steps==0] = -1
    if sync_jitters:
        foreground_steps = background_steps 
        foreground_velocity = background_velocity
    else:
        foreground_steps = np.round(np.random.random(n_frames))
        foreground_steps = foreground_steps.astype(np.int)
        foreground_steps[foreground_steps==0] = -1
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                     image_shape)
    frames = []
    for i in range(n_frames):
        step = np.random.random(1) < background_velocity
        if step:
            background_grating = np.roll(background_grating,
                                        background_steps[i],
                                        axis=background_axis)
        new_frame = background_grating.copy()
            
        step = np.random.random(1) < foreground_velocity
        if step:
            circle_grating = np.roll(circle_grating,
                                foreground_steps[i],
                                axis=foreground_axis)

        temp = circle_grating[row_coords,col_coords]
        new_frame[row_coords,col_coords] = temp
        frames.append(new_frame)
    return np.asarray(frames), background_grating, circle_grating

def differential_circle(n_frames=100, bar_width=4,
                                       inner_bar_width=None,
                                       foreground_velocity=0.5,
                                       background_velocity=0,
                                       image_shape=(50,50),
                                       center=(25,25),
                                       radius=5,
                                       init_offset=0,
                                       horizontal_foreground=False,
                                       horizontal_background=False,
                                       background_grating=None,
                                       circle_grating=None):
    """
    Creates circle window onto a grating that has stripes
    perpendicular to the background.  The grating behind this window
    then rolls differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_width: int
            the size of the grating bars in pixels
        inner_bar_width: int
            size of grating bars inside circle. If None, set to
            bar_width
        foreground_velocity: float
            the foreground (grating circle) moves to the right with
            positive velocities and left with negative. units of
            pixels per frame
        background_velocity: float
            the background (grating) moves to the right with positive
            velocities and left with negative. units of pixels per
            frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is
            the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        init_offset: int
            the initial offset of the foreground grating from the
            background grating
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical
        background_grating: ndarray (H,W)
            optionally pass custom background image through
        circle_grating: ndarray (H,W)
            optionally pass custom foreground image through

    Returns:
        frame sequence of shape (n_frames, image_shape[0],
                                           image_shape[1])
    """
    bar_width=int(bar_width)
    if inner_bar_width is None: 
        inner_bar_width = bar_width
    else:
        inner_bar_width = int(inner_bar_width)
    background_axis = 0 if horizontal_background else 1
    foreground_axis = 0 if horizontal_foreground else 1
    if circle_grating is None:
        angle = 0 if horizontal_foreground else 90
        circle_grating = get_striped_img(image_shape, inner_bar_width,
                                                         angle=angle)
    circle_grating = np.roll(circle_grating, init_offset,
                                    axis=foreground_axis)
    if background_grating is None:
        angle = 0 if horizontal_background else 90
        background_grating = get_striped_img(image_shape, bar_width,
                                                        angle=angle)
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                     image_shape)
    frames = []
    for i in range(n_frames):
        background_dist = int(background_velocity*i)
        if background_dist != 0:
            new_frame = np.roll(background_grating, background_dist,
                                               axis=background_axis)
        else:
            new_frame = background_grating.copy()
            
        foreground_dist = int(foreground_velocity*i)
        if foreground_dist != 0:
            new_circ = np.roll(circle_grating, foreground_dist,
                                          axis=foreground_axis)
        else:
            new_circ = circle_grating.copy()
        new_frame[row_coords, col_coords] = new_circ[row_coords,
                                                     col_coords]
        frames.append(new_frame)
    return np.asarray(frames), background_grating, circle_grating

def jittered_circle(n_frames=100, bar_width=4, inner_bar_width=None,
                                         foreground_jitter=0.5,
                                         background_jitter=0,
                                         image_shape=(50,50),
                                         center=(25,25),
                                         radius=5,
                                         horizontal_foreground=False,
                                         horizontal_background=False,
                                         step_size=1,
                                         background_grating=None,
                                         circle_grating=None):
    """
    Creates circle window onto a grating that has stripes
    perpendicular to the background. This window then jitters
    differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_width: int
            the size of the grating bars in pixels
        inner_bar_width: int
            size of grating bars inside circle. If None, set to
            bar_width
        foreground_jitter: positive float (between 0 and 1)
            the foreground (grating circle) jitter frequency.
            larger number means more jitters per second
        background_jitter: positive float (between 0 and 1)
            the background (grating) jitter intensity.
            larger number means more jitters per second
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is
            the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical
        step_size: positive int
            the largest magnitude possible jitter distance in pixels
        background_grating: ndarray (H,W)
            optionally pass custom background image through
        circle_grating: ndarray (H,W)
            optionally pass custom foreground image through

    Returns:
        frame sequence of shape (n_frames, image_shape[0],
                                                    image_shape[1])

    """
    bar_width = int(bar_width)
    if inner_bar_width is None: 
        inner_bar_width = bar_width
    else:
        inner_bar_width = int(inner_bar_width)
    foreground_jitter = float(np.clip(abs(foreground_jitter), 0, 1))
    background_jitter = float(np.clip(abs(background_jitter), 0, 1))
    step_size = int(max(abs(step_size),1))
    background_axis = 0 if horizontal_background else 1
    foreground_axis = 0 if horizontal_foreground else 1
    if circle_grating is None:
        angle = 0 if horizontal_foreground else 90
        circle_grating = get_striped_img(image_shape, inner_bar_width,
                                                         angle=angle)
    # Roll to start with unaligned gratings
    circle_grating = np.roll(circle_grating, 2, axis=foreground_axis)
    # Make random center shifts
    row_shifts = np.random.randint(-step_size,step_size+1, n_frames)
    # Make random center shifts
    col_shifts = np.random.randint(-step_size,step_size+1, n_frames)
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                     image_shape)

    if background_grating is None:
        angle = 0 if horizontal_background else 90
        background_grating = get_striped_img(image_shape, bar_width,
                                                        angle=angle)
    # Make random background shifts
    background_shifts = np.random.randint(-1, 2, n_frames)

    frames = []
    for i in range(n_frames):
        shift_idx = int(background_jitter*i)
        if shift_idx != 0:
            background_grating = np.roll(background_grating,
                                background_shifts[shift_idx],
                                axis=background_axis)
            background_shifts[shift_idx] = 0
        new_frame = background_grating.copy()
            
        shift_idx = int(foreground_jitter*i)
        if shift_idx != 0:
            center = (center[0]+row_shifts[shift_idx],
                              center[1]+col_shifts[shift_idx])
            row_coords, col_coords = skimage.draw.circle(*center,
                                                         radius,
                                                         image_shape)
            row_shifts[shift_idx], col_shifts[shift_idx] = 0, 0
        new_frame[row_coords, col_coords] = circle_grating[row_coords,
                                                           col_coords]
        frames.append(new_frame)
    return np.asarray(frames), background_grating, circle_grating

def moving_circle(n_frames=100, bar_width=4, inner_bar_width=None,
                                        foreground_velocity=0.5,
                                        background_velocity=0,
                                        image_shape=(50,50),
                                        center=(25,25),
                                        radius=5,
                                        horizontal_background=False,
                                        backround_grating=None,
                                        circle_grating=None):
    """
    Creates circle window onto a grating that has stripes
    perpendicular to the background. This window then translates
    differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_width: int
            the size of the grating bars in pixels
        inner_bar_width: int
            size of grating bars inside circle. If None, set to
            bar_width
        foreground_velocity: float
            the foreground (grating circle) moves to the right with
            positive velocities and left with negative. units of
            pixels per frame
        background_velocity: float
            the background (grating) moves to the right with positive
            velocities and left with negative. units of pixels per
            frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is
            the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical
        background_grating: ndarray (H,W)
            optionally pass custom background image through
        circle_grating: ndarray (H,W)
            optionally pass custom foreground image through

    Returns:
        frame sequence of shape (n_frames, image_shape[0],
                                                image_shape[1])
    """
    bar_width=int(bar_width)
    if inner_bar_width is None: 
        inner_bar_width = bar_width
    else:
        inner_bar_width = int(inner_bar_width)
    angle = 90 if horizontal_background else 0
    if circle_grating is None:
        circle_grating = get_striped_img(image_shape, inner_bar_width,
                                                         angle=angle)
    angle = (angle+90)%180
    if background_grating is None:
        background_grating = get_striped_img(image_shape, bar_width,
                                                        angle=angle)
    row_coords, col_coords = skimage.draw.circle(*center, radius,
                                                     image_shape)
    frames = []
    for i in range(n_frames):
        background_dist = int(background_velocity*i)
        if background_dist != 0:
            new_frame = shift(background_grating, background_fill=0,
                                          shift=(0,background_dist))
        else:
            new_frame = background_grating.copy()
            
        foreground_dist = int(foreground_velocity*i)
        if foreground_dist != 0:
            new_center = (center[0], center[1]+foreground_dist)
            tup = skimage.draw.circle(*new_center, radius,
                                              image_shape)
            new_row_coords, new_col_coords = tup
        else:
            new_row_coords, new_col_coords = row_coords, col_coords
        s = np.s_[new_row_coords, new_col_coords]
        new_frame[s] = circle_grating[s]
        frames.append(new_frame)
    return np.asarray(frames), background_grating, circle_grating

def shift(img, background_fill=0, shift=(0,0)):
    """
    Shifts an image by the specified amount. The remaining null space
    is filled with the background fill.

    img - the image to be shifted shape (H,W)
    background_fill - the value to fill the null space following the
        shift
    shift - tuple of shift amounts. (vertical shift, horizontal shift)
        a positive vertical shift shifts up, negative shifts down
        a positive horizontal shift shifts right, negative shifts left
    """
    M = np.float32([[1,0,shift[1]],
                    [0,1,shift[0]]])
    rows, cols = img.shape
    shifted = cv2.warpAffine(img, M, (cols,rows),
                     borderValue=background_fill)
    return shifted

def unroll(X):
    """Unrolls a toeplitz stimulus"""
    return np.vstack((X[0, :], X[:, -1]))

def spatial_pad(stimulus, H, W=None):
    """
    Pads the stimulus with zeros up to H and W

    stimulus - ndarray (..., H1, W1)
    H - int
        final desired spatial height of stimulus
    W - int
        final desired spatial width of stimulus. defaults to H if None
    """
    if W is None:
        W = H
    if stimulus.shape[-2] >= H and stimulus.shape[-1] >= W:
        return stimulus

    pad_h = max(0,(H-stimulus.shape[-2]))
    pad_w = max(0,(W-stimulus.shape[-1]))
    pad_h = (pad_h//2,pad_h//2+(pad_h%2))
    pad_w = (pad_w//2,pad_w//2+(pad_w%2))
    if type(stimulus) == type(torch.FloatTensor()):
        leading_zeros = [0 for i in range((len(stimulus.shape)-2)*2)]
        pads = (*pad_w, *pad_h, *leading_zeros)
        return F.pad(stimulus, pads, "constant", 0)
    else:
        leading_zeros = [(0,0) for i in range(len(stimulus.shape)-2)]
        p = np.pad(stimulus,(*leading_zeros,pad_h,pad_w),"constant")
        return p

def shifted_overlay(img, shift_img, row_shift=0, col_shift=0):
    """
    Pastes the shift_img onto the img at the shift coordinates.
    0,0 corresponds to the center of the image.

    WARNING: OPERATES INPLACE ON ARGUED IMG

    img: ndarray (...,H,W)
    shift_img: ndarray (...,H',W')
    row_shift: int
        number of pixels shifted in the 0th dimension
    col_shift: int
        number of pixels shifted in the 1st dimension

    Returns:
        combined_img: ndarray (H,W)
            img with shift_img overlayed at the shifted location
    """
    shift_rows = shift_img.shape[-2]
    shift_cols = shift_img.shape[-1]
    img_rows = img.shape[-2]
    img_cols = img.shape[-1]
    
    img_half = img_rows//2
    shift_half = shift_rows//2
    row_lim = min(shift_rows, img_rows)+abs(img_half-shift_half)
    img_half = img_cols//2
    shift_half = shift_cols//2
    col_lim = min(shift_cols, img_cols)+abs(img_half-shift_half)
    if abs(row_shift) >= row_lim or abs(col_shift) >= col_lim:
        return img

    # Get the shift cut
    # Rows
    low_row_diff = shift_rows//2-row_shift-img_rows//2
    row_low = max(0,low_row_diff)
    odd_shift = (shift_rows//2)*2 != shift_rows
    odd_img = (img_rows//2)*2!=img_rows
    high_row_diff = shift_rows//2+odd_shift+row_shift-\
                                  (img_rows//2+odd_img)
    row_high = min(shift_rows, shift_rows-high_row_diff)
    # Cols
    low_col_diff = shift_cols//2-col_shift-img_cols//2
    col_low = max(0,low_col_diff)
    odd_shift = (shift_cols//2)*2 != shift_cols
    odd_img = (img_cols//2)*2!=img_cols
    high_col_diff = shift_cols//2+odd_shift+col_shift-\
                                  (img_cols//2+odd_img)
    col_high = min(shift_cols, shift_cols-high_col_diff)
    
    shift_cut = shift_img[...,row_low:row_high, col_low:col_high]

    # Get the img cut
    row_low = max(0,-low_row_diff)
    row_high = min(img_rows, img_rows+high_row_diff)
    col_low = max(0,-low_col_diff)
    col_high = min(img_cols, img_cols+high_col_diff)
    
    img[...,row_low:row_high,col_low:col_high] = shift_cut
    return img

def prepad(y, n=40, v=0):
    """
    prepads with value
    """
    pad = v * np.ones((n, y.shape[1]))
    return np.vstack((pad, y))

def concat(*args, nx=50, nh=40):
    """
    Returns a spatiotemporal stimulus that has been transformed using
    rolling_window given a list of stimuli to concatenate

    Parameters
    ----------
    stimuli : iterable
        A list or iterable of stimuli (numpy arrays). The first
        dimension is the sample, which can be different, but the
        rest of the dimensions (spatial dimensions) must be the same

    nh : int, optional
        Number of time steps in the rolling window history
        (default: 40)

    nx : int, optional
        Number of spatial dimensions (default: 50)
    """
    c = np.vstack(map(lambda s: spatialize(s, nx), args))
    c = c.astype('float32')
    return rolling_window(c, nh)

def white(nt, nx=1, contrast=1.0):
    """
    Gaussian white noise with the given contrast. The white noise is
    spatially homogeneous.

    Parameters
    ----------
    nt : int
        number of temporal samples

    nx : int
        number of spatial dimensions (default: 1)

    contrast : float
        Scalar multiplied by the whole stimulus (default: 1.0)
    """
    return spatialize(contrast * np.random.randn(nt, 1, 1), nx=nx)

def repeat_white(n_samples, nx=50, contrast=1.0, n_repeats=3,
                                            rand_spat=False):
    """
    Creates and returns a stimulus in which the screen that can be
    spatially homogeneous or heterogeneous, and repeats temporally
    for every n_repeats.

    n_samples: int
        the length of the stimulus in samples 
    nx: int
        size of height and width of the stimulus
    contrast: float
        standard deviation of the flickering
    n_repeats: int
        the number of frames to repeat before switching to the next
        intensity
    rand_spat: bool
        if true, the spatial dimensions are also randomized. If
        false, all spatial values are the same for each frame in
        time.
    """
    compressed_time = int(np.ceil(n_samples/n_repeats))
    if rand_spat:
        compressed_stim  = np.random.randn(compressed_time, nx, nx)*contrast
    else:
        compressed_stim = white(compressed_time, nx=nx,
                                            contrast=contrast)
    stimulus = np.repeat(compressed_stim, n_repeats, axis=0)
    return stimulus[:n_samples]

def dir_selectivity(img_shape=(50,50), bar_width=20, angle=0,
                                                step_size=10,
                                                n_repeats=3):
    """
    Direction selectivity stimulus

    img_shape: tuple of ints (height,width)
    bar_width: int
        size of moving bar
    angle: int (0-360)
        the angle of the bar from upwards from the rightward
        horizontal
    step_size: int
        the movement of the bar in terms of pixels per frame
    n_repeats: int
        the number of frames to repeat each of the images
    """
    frames = tdrstim.dir_select_vid(img_shape, bar_width=bar_width,
                                  angle=angle, step_size=step_size)
    ones = [1 for i in range(len(frames.shape[1:]))]
    return np.tile(frames, (n_repeats, *ones))

def contrast_steps(contrasts, lengths, nx=1):
    """
    Returns a random sequence with changing contrasts. A contrast
    is used for its corresponding `length`. A more general way to 
    describe this function is a sequence of whitenoise with differing
    chunks of standard deviations. Each standard deviation (contrast)
    is used for the specified length (number of frames).

    Parameters
    ----------
    contrasts : array_like
        List of the contrasts in the sequence

    lengths : int or array_like
        If an integer is given, each sequence has the same length.
        Otherwise, the given list is used as the lengths for each
        contrast

    nx : int
        Number of spatial dimensions (default: 1)
    """
    if isinstance(lengths, int):
        lengths = repeat(lengths)

    return np.vstack([white(nt, nx=nx, contrast=sigma)
                      for sigma, nt in zip(contrasts, lengths)])


def spatialize(array, nx):
    """Returns a spatiotemporal version of a full field stimulus

    Given an input array of shape (t, 1, 1), returns a new array with
    shape (t, nx, nx) where each temporal value is copied at each
    spatial location

    Parameters
    ----------
    array : array_like
        The full-field stimulus to spatialize

    nx : int
        The number of desired spatial dimensions (along one edge)
    """
    if nx % array.shape[1] == 0:
        return np.broadcast_to(array, (array.shape[0], nx, nx))
    else:
        return spatial_pad(array,nx)


def flash(flash_dur, delay, nsamples, intensity=-1.):
    """Generates a 1D flash

    Parameters
    ----------
    flash_dur : int
        The duration (in samples) of the flash

    delay : int
        The delay (in samples) before the flash starts

    nsamples : int
        The total number of samples in the array

    intensity : float or array_like, optional
        The flash intensity. If a number is given, the flash is a
        full-field flash. Otherwise, if it is a 2D array, then that
        image is flashed. (default: 1.0)
    """
    # generate the image to flash
    if isinstance(intensity, Number):
        img = intensity * np.ones((1, 1, 1))
    else:
        img = intensity.reshape(1, *intensity.shape)

    assert nsamples >= (delay + flash_dur), \
        "The total number samples must be greater than the delay +\
                                                        flash_dur"
    sequence = np.zeros((nsamples,))
    sequence[delay:(delay + flash_dur)] = 1.0
    return sequence.reshape(-1, 1, 1) * img


def bar(center, width, height, nx=50, intensity=-1., us_factor=1,
                                                        blur=0.):
    """
    Create a single frame of with a bar drawn at the argued center.
    """

    # upscale factor (for interpolation between discrete bar
    # locations)
    c0 = center[0] * us_factor
    c1 = center[1] * us_factor
    width *= us_factor
    height *= us_factor
    nx *= us_factor

    # center of the bar
    cx, cy = int(c0 + nx // 2), int(c1 + nx // 2)

    # x- and y- indices of the bar
    bx = slice(max(0, cx - width // 2),
                        max(0, min(nx, cx + width // 2)))
    by = slice(max(0, cy - height // 2),
                        max(0, min(nx, cy + height // 2)))

    # set the bar intensity values
    frame = np.zeros((nx, nx))
    frame[by, bx] = intensity

    # downsample the blurred image back to the original size
    return downsample(frame, us_factor, blur)
    
def downsample(img, factor, blur):
    """Smooth and downsample the image by the given factor"""
    return downscale_local_mean(gaussian(img, blur), (factor,
                                                     factor))


def driftingbar(velocity, width, intensity=-1., x=(-30, 30)):
    """Drifting bar

    Usage
    -----
    >>> centers, stim = driftingbar(0.08, 2)

    Parameters
    ----------
    velocity : float
        bar velocity in pixels / frame (if negative, the bar reverses
                                                           direction)

    width : int
        bar width in pixels

    Returns
    -------
    centers : array_like
        The center positions of the bar at each frame in the stimulus

    stim : array_like
        The spatiotemporal drifting bar movie
    """
    npts = 1 + abs(int((x[1] - x[0]) / velocity))
    centers = np.linspace(x[0], x[1], npts)
    fxn = lambda x: bar((x, 0), width, np.Inf, us_factor=5, blur=0.)
    c = np.stack(map(fxn, centers))
    c = concat(c)
    return centers, velocity, c


def cmask(center, radius, array):
    """
    Generates a mask covering a central circular region
    """
    a, b = center
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    return x ** 2 + y ** 2 <= radius ** 2


def paired_flashes(ifi, flash_dur, intensity, total_length, delay):
    """Paired flash stimulus"""
    # Convert numbers to tuples
    flash_dur = tuplify(flash_dur, 2)
    intensity = tuplify(intensity, 2)

    # generate the flashes
    f0 = flash(flash_dur[0], delay, total_length, intensity[0])
    f1 = flash(flash_dur[1], delay + flash_dur[0] + ifi, total_length,
                                                         intensity[1])

    # return the concatenated pair
    return concat(f0 + f1)

def sinusoid(halfperiod, nsamples, phase=0., intensity=1.0):
    """Generates a 1-D sinusoidal wave"""
    assert 0 <= phase <= 1, "Phase must be a fraction between 0 and 1"

    # if halfperiod is zero, return all ones
    if halfperiod == 0:
        return np.ones(nsamples)

    # compute the base frequency
    time = np.linspace(0, 1, nsamples)
    freq = 1 / (2 * halfperiod / nsamples)

    # convert to radians
    omega = 2 * np.pi * freq
    phi = 2 * np.pi * phase

    return intensity * np.sin(omega * time + phi)


def square(halfperiod, nsamples, phase=0., intensity=1.0):
    """Generates a 1-D square wave"""
    assert 0 <= phase <= 1, "Phase must be a fraction between 0 and 1"

    # if halfperiod is zero, return all ones
    if halfperiod == 0:
        return np.ones(nsamples)

    # discretize the offset in terms of the period
    offset = int(2 * phase * halfperiod)

    # generate one period of the waveform
    waveform = np.stack(repeat(np.array([intensity, -intensity]),
                                           halfperiod)).T.ravel()

    # generate the repeated sequence
    repeats = int(np.ceil(nsamples / (2 * halfperiod)) + 1)
    sequence = np.hstack(repeat(waveform, repeats))

    # use the offset to specify the phase
    return sequence[offset:(nsamples + offset)]


def grating(barsize=(5, 0), phase=(0., 0.), nx=50,
                        intensity=(1., 1.), us_factor=1,
                        blur=0., waveform='square'):
    """Returns a grating as a spatial frame

    Parameters
    ----------
    barsize : (int, int), optional
        Size of the bar in the x- and y- dimensions. A size of 0 indicates no spatial
        variation along that dimension. Default: (5, 0)

    phase : (float, float), optional
        The phase of the grating in the x- and y- dimensions (as a fraction of the period).
        Must be between 0 and 1. Default: (0., 0.)

    intensity=(1., 1.)
        The contrast of the grating for the x- and y- dimensions

    nx : int, optional
        The number of pixels along each dimension of the stimulus (default: 50)

    us_factor : int
        Amount to upsample the image by (before downsampling back to 50x50), (default: 1)

    blur : float
        Amount of blur to applied to the upsampled image (before downsampling), (default: 0.)

    waveform : str
        Either 'square' or 'sin' or 'sinusoid'
    """
    if waveform == 'square':
        wform = square
    elif waveform in ('sin', 'sinusoid'):
        wform = sinusoid
    else:
        raise ValueError('Invalid waveform: {}.'.format(waveform))

    # generate a square wave along each axis
    x = wform(barsize[0], nx * us_factor, phase[0], intensity[0])
    y = wform(barsize[1], nx * us_factor, phase[1], intensity[1])

    # generate the grating frame and downsample
    return downsample(np.outer(y, x), us_factor, blur)


def jittered_grating(nsamples, sigma=0.1, size=3):
    """
    Creates a grating that jitters over time according to a
    random walk
    """
    phases = np.cumsum(sigma * np.random.randn(nsamples)) % 1.0
    frames = np.stack([grating(barsize=(size, 0), phase=(p, 0.))\
                                                for p in phases])
    return frames


def drifting_grating(nsamples, dt, barsize, us_factor=1, blur=0.):
    """Generates a drifting vertical grating

    Parameters
    ----------
    nsamples : int
        The total number of temporal samples

    dt : float
        The timestep of each sample. A smaller value of dt will
        generate a slower drift

    barsize : int
        The width of the bar in samples

    us_factor : int, optional
        Amount to upsample the image by (before downsampling back to
        50x50), (default: 1)

    blur : float, optional
        Amount of blur to applied to the upsampled image (before
        downsampling), (default: 0.)
    """
    phases = np.mod(np.arange(nsamples) * dt, 1)
    return np.stack([grating(barsize=(barsize, 0),
                             phase=(phi, 0.),
                             us_factor=us_factor,
                             blur=blur) for phi in phases])


def reverse(img, halfperiod, nsamples):
    """
    Generates a temporally reversing stimulus using the given image

    Parameters
    ----------
    img : array_like
        A spatial image to reverse (e.g. a grating)

    halfperiod : int
        The number of frames each half period of the reversing image
        is shown for

    nsamples : int
        The total length of the stimulus in samples
    """
    return np.stack([t * img for t in square(halfperiod, nsamples)])


def get_grating_movie(grating_width=1, switch_every=10,
                        movie_duration=100, mask=False,
                      intensity=1, phase=0, roll=True):
    '''
    Returns a reversing gratings stimulus.

    INPUT:
        grating_width   the width (in checkers) of each horizontal bar
        switch_every    how often (in frames) you want to reverse 
                        grating polarity
        movie_duration  number of frames in stimulus
        mask            either False or a np.array of shape (50,50);
                        masks the gratings (i.e., over the receptive
                        field)
        intensity       what is the contrast of the gratings?
        phase           what is the phase of the gratings?

    OUTPUT:
        full_movies     an np.array of shape
                                        (movie_duration, 40, 50, 50)
    '''

    # make grating
    grating_frame = -1 * np.ones((50, 50))
    for i in range(grating_width):
        grating_frame[:, (i + phase)::2 * grating_width] = 1
    if mask:
        grating_frame = grating_frame * mask * intensity
    else:
        grating_frame = grating_frame * intensity

    # make movie
    grating_movie = np.zeros((movie_duration, 50, 50))
    polarity_count = 0
    for frame in range(movie_duration):
        polarity_count += 1
        if int(polarity_count / switch_every) % 2 == 0:
            grating_movie[frame] = grating_frame
        else:
            grating_movie[frame] = -1 * grating_frame

    if roll:
        # roll movie axes to get the right shape
        full_movies = rolling_window(grating_movie, 40)
        return full_movies
    else:
        return grating_movie

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1
        (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1\
                                                              (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape,
                                             strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr


def tuplify(x, n):
    """
    Converts a number into a tuple with that number repeating

    Usage
    -----
    >>> tuplify(3, 5)
    (3, 3, 3, 3, 3)
    >>> tuplify((1,2), 5)
    (1, 2)
    """
    if isinstance(x, Number):
        x = tuple(repeat(x, n))
    return x
