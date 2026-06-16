"""
face_crop.py
------------
Shared face detection and cropping utility.
Uses OpenCV's built-in Haar cascade — no extra downloads needed.

Used by: finetune_models.py, mesonet_inference.py, xception_inference.py

HOW IT WORKS:
  1. Detect faces in the frame using Haar cascade
  2. If a face is found, crop it with a 30% margin and return
  3. If no face is found, return the centre crop of the frame
     (fallback ensures no video is skipped entirely)
"""

import cv2
import numpy as np

# Load cascade once at import time
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(frame_rgb, margin=0.3, min_size=64):
    """
    Detect and crop the largest face in a frame.

    Args:
        frame_rgb : RGB numpy array (H, W, 3)
        margin    : fractional padding around detected face box
        min_size  : minimum face size in pixels to consider

    Returns:
        Cropped RGB numpy array of the face region.
        Falls back to centre crop if no face detected.
    """
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w  = frame_rgb.shape[:2]

    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (min_size, min_size),
        flags        = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Fallback: return centre crop (60% of frame)
        ch, cw = int(h * 0.2), int(w * 0.2)
        return frame_rgb[ch:h-ch, cw:w-cw]

    # Pick the largest detected face
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Add margin
    mx = int(fw * margin)
    my = int(fh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)

    return frame_rgb[y1:y2, x1:x2]


def extract_frames_with_faces(video_path, num_frames=10):
    """
    Sample frames from a video and return face-cropped versions.

    Returns list of RGB face crop arrays.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    crops   = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face      = extract_face(frame_rgb)
            if face.size > 0:
                crops.append(face)

    cap.release()
    return crops
