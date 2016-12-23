import os

import cv2
import numpy as np

# import code # for debug
#use in detection will migrate to opencv
import imutils


def face_detect(this_Frame, face_cascade, profileface_cascade, window = None):
    """Face detection based on Python-OpenCV2.

    Parameters
    ------------
    this_Frame : numpy array
        [row, col, 3]; 3 means [B, G, R]
    face_cascade : OpenCV object for detecting face from front
    profileface_cascade : OpenCV object for detecting face from side
    window : tuple of int or None (default)
        Bounding box for detection, None means detect from the whole frame.

    Returns
    -------
    list of bounding box, [(x,y,w,h), (x,y,w,h), (x,y,w,h) ...]

    Examples
    ----------
    >>>
    >>>
    """
    if window is None:
        gray = cv2.cvtColor(this_Frame, cv2.COLOR_BGR2GRAY)
    else:
        (x0,y0,w0,h0) = window
        gray = cv2.cvtColor(this_Frame[y0:y0+h0,x0:x0+w0,:], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 4) #1.3
    profiles = profileface_cascade.detectMultiScale(gray, 1.4, 5)

    if len(faces)>0:
        if len(profiles)>0:
            return np.vstack((faces,profiles))
        else:
            return faces
    else:
        return profiles
