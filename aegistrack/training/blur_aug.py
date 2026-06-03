from __future__ import annotations

import math
import random

import cv2
import numpy as np

import albumentations as A


def _odd(v: int) -> int:
    v = max(3, int(v))
    return v if v % 2 == 1 else v + 1


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = _odd(length)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0

    center = (length / 2.0 - 0.5, length / 2.0 - 0.5)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (length, length), flags=cv2.INTER_LINEAR)

    s = float(kernel.sum())
    if s <= 1e-6:
        kernel[length // 2, length // 2] = 1.0
        s = 1.0

    return kernel / s


def _sample_blur_angle(cfg) -> float:
    r = random.random()
    hp = float(getattr(cfg, "blur_horizontal_prob", 0.35))
    vp = float(getattr(cfg, "blur_vertical_prob", 0.35))
    dp = float(getattr(cfg, "blur_diagonal_prob", 0.20))
    jitter = float(getattr(cfg, "blur_angle_jitter_deg", 10.0))

    if r < hp:
        base = random.choice([0.0, 180.0])
    elif r < hp + vp:
        base = random.choice([90.0, -90.0])
    elif r < hp + vp + dp:
        base = random.choice([45.0, -45.0, 135.0, -135.0])
    else:
        base = random.uniform(-180.0, 180.0)

    return base + random.uniform(-jitter, jitter)


def _clip_bbox_xywh(bbox, image_w: int, image_h: int):
    x, y, w, h = map(float, bbox)
    x1 = max(0.0, min(float(image_w - 1), x))
    y1 = max(0.0, min(float(image_h - 1), y))
    x2 = max(0.0, min(float(image_w), x + max(1.0, w)))
    y2 = max(0.0, min(float(image_h), y + max(1.0, h)))

    if x2 <= x1:
        x2 = min(float(image_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h), y1 + 1.0)

    return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)


def _expand_bbox_along_angle(bbox, angle_deg: float, kernel_len: int, image_w: int, image_h: int, cfg):
    x, y, w, h = map(float, bbox)
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    theta = math.radians(angle_deg)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))

    expand_parallel = min(
        float(getattr(cfg, "blur_bbox_expand_parallel_factor", 0.30)) * float(kernel_len),
        float(getattr(cfg, "blur_bbox_expand_parallel_max_px", 6.0)),
    )
    expand_perp = float(getattr(cfg, "blur_bbox_expand_perp_px", 1.0))
    max_ratio = float(getattr(cfg, "blur_aug_max_bbox_expand_ratio", 1.6))

    add_w = 2.0 * (expand_parallel * c + expand_perp * s)
    add_h = 2.0 * (expand_parallel * s + expand_perp * c)

    new_w = min(w + add_w, max(w * max_ratio, w + 1.0))
    new_h = min(h + add_h, max(h * max_ratio, h + 1.0))

    return _clip_bbox_xywh(
        (cx - 0.5 * new_w, cy - 0.5 * new_h, new_w, new_h),
        image_w,
        image_h,
    )


def _gaussian_blur(image: np.ndarray, kernel_len: int) -> np.ndarray:
    if A is not None:
        aug = A.GaussianBlur(blur_limit=(kernel_len, kernel_len), p=1.0)
        return aug(image=image)["image"]
    return cv2.GaussianBlur(image, (kernel_len, kernel_len), 0)


def apply_train_blur_stretch(search_img, search_bbox, cfg):
    if not bool(getattr(cfg, "blur_aug_enabled", False)):
        return search_img, search_bbox

    if random.random() >= float(getattr(cfg, "blur_aug_prob", 0.15)):
        return search_img, search_bbox

    image_h, image_w = search_img.shape[:2]

    angle = _sample_blur_angle(cfg)
    kernel_len = random.randint(
        int(getattr(cfg, "motion_blur_kernel_min", 3)),
        int(getattr(cfg, "motion_blur_kernel_max", 11)),
    )
    kernel_len = _odd(kernel_len)

    if random.random() < float(getattr(cfg, "motion_blur_prob", 0.75)):
        kernel = _motion_blur_kernel(kernel_len, angle)
        search_img = cv2.filter2D(search_img, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    else:
        search_img = _gaussian_blur(search_img, kernel_len)

    search_bbox = _expand_bbox_along_angle(
        search_bbox,
        angle_deg=angle,
        kernel_len=kernel_len,
        image_w=image_w,
        image_h=image_h,
        cfg=cfg,
    )

    return search_img, search_bbox
