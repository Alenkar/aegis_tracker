from __future__ import annotations
import random
import cv2
import numpy as np


def tiny_uav_augment(img, blur: bool = True):
    out = img.copy()
    if blur and random.random() < 0.35:
        k = random.choice([3, 5, 7, 9])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.40:
        noise = np.random.normal(0, random.uniform(2, 12), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.40:
        alpha = random.uniform(0.6, 1.4)
        beta = random.uniform(-25, 25)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        q = random.randint(25, 80)
        _, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out
