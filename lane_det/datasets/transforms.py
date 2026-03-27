import cv2
import numpy as np


def _parse_range(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (int, float)):
        value = float(value)
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected range with 2 values, got: {value}")
    return (float(value[0]), float(value[1]))


def build_translate_shear_matrix(img_h, tx=0.0, ty=0.0, shear_x=0.0):
    center_y = 0.5 * max(float(img_h - 1), 1.0)
    return np.array(
        [
            [1.0, float(shear_x), float(tx - shear_x * center_y)],
            [0.0, 1.0, float(ty)],
        ],
        dtype=np.float32,
    )


def warp_lane_rows(lanes, valid_mask, h_samples, matrix, img_w, img_h):
    if lanes is None or valid_mask is None or h_samples is None:
        return lanes, valid_mask

    lanes = np.asarray(lanes, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)
    h_samples = np.asarray(h_samples, dtype=np.float32)

    warped_lanes = np.zeros_like(lanes, dtype=np.float32)
    warped_mask = np.zeros_like(valid_mask, dtype=np.uint8)

    if lanes.shape[0] == 0 or h_samples.size == 0:
        return warped_lanes, warped_mask

    for lane_idx in range(lanes.shape[0]):
        valid_idx = np.flatnonzero(valid_mask[lane_idx] > 0)
        if valid_idx.size == 0:
            continue

        points = np.stack(
            [lanes[lane_idx, valid_idx], h_samples[valid_idx]],
            axis=1,
        ).astype(np.float32)
        warped_points = cv2.transform(points[None, :, :], matrix)[0]
        x_coords = warped_points[:, 0]
        y_coords = warped_points[:, 1]

        order = np.argsort(y_coords)
        x_coords = x_coords[order]
        y_coords = y_coords[order]

        rounded_y = np.round(y_coords, decimals=4)
        _, unique_indices = np.unique(rounded_y, return_index=True)
        x_coords = x_coords[unique_indices]
        y_coords = y_coords[unique_indices]

        if y_coords.size == 1:
            target_idx = int(np.argmin(np.abs(h_samples - y_coords[0])))
            if 0.0 <= x_coords[0] <= (img_w - 1.0) and 0.0 <= y_coords[0] <= (img_h - 1.0):
                warped_lanes[lane_idx, target_idx] = x_coords[0]
                warped_mask[lane_idx, target_idx] = 1
            continue

        x_interp = np.interp(h_samples, y_coords, x_coords, left=np.nan, right=np.nan)
        interp_mask = np.isfinite(x_interp)
        interp_mask &= h_samples >= y_coords[0]
        interp_mask &= h_samples <= y_coords[-1]
        interp_mask &= x_interp >= 0.0
        interp_mask &= x_interp <= (img_w - 1.0)

        warped_lanes[lane_idx, interp_mask] = x_interp[interp_mask].astype(np.float32)
        warped_mask[lane_idx, interp_mask] = 1

    return warped_lanes, warped_mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        for t in self.transforms:
            image, lanes, valid_mask, h_samples = t(image, lanes, valid_mask, h_samples)
        return image, lanes, valid_mask, h_samples


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() < self.p:
            h, w = image.shape[:2]
            image = image[:, ::-1, :].copy()
            if lanes is not None and valid_mask is not None:
                lanes = lanes.copy()
                flip_mask = valid_mask > 0
                lanes[flip_mask] = (w - 1) - lanes[flip_mask]
        return image, lanes, valid_mask, h_samples


class ColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() >= self.p:
            return image, lanes, valid_mask, h_samples

        if np.random.random() < 0.5:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if np.random.random() < 0.5:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean_val = image.mean()
            image = np.clip((image - mean_val) * factor + mean_val, 0, 255).astype(np.uint8)

        if np.random.random() < 0.5:
            # Note: TuSimple loader uses cv2.cvtColor(BGR2RGB), so image is RGB
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image, lanes, valid_mask, h_samples


class RandomOcclusion:
    def __init__(
        self,
        p=0.0,
        num_range=(1, 2),
        width_range=(0.08, 0.18),
        height_range=(0.08, 0.22),
    ):
        self.p = float(p)
        self.num_range = tuple(int(v) for v in num_range)
        self.width_range = _parse_range(width_range, (0.08, 0.18))
        self.height_range = _parse_range(height_range, (0.08, 0.22))

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() >= self.p:
            return image, lanes, valid_mask, h_samples

        image = image.copy()
        img_h, img_w = image.shape[:2]
        min_count, max_count = self.num_range
        num_blocks = int(np.random.randint(min_count, max_count + 1))

        for _ in range(max(num_blocks, 1)):
            occ_w = max(1, int(round(np.random.uniform(*self.width_range) * img_w)))
            occ_h = max(1, int(round(np.random.uniform(*self.height_range) * img_h)))
            x0 = int(np.random.randint(0, max(img_w - occ_w + 1, 1)))
            y0 = int(np.random.randint(0, max(img_h - occ_h + 1, 1)))
            fill_color = np.random.randint(0, 160, size=(1, 1, 3), dtype=np.uint8)
            image[y0 : y0 + occ_h, x0 : x0 + occ_w] = fill_color

        return image, lanes, valid_mask, h_samples


class RandomShadow:
    def __init__(
        self,
        p=0.0,
        width_range=(0.2, 0.5),
        height_range=(0.2, 0.45),
        darkness_range=(0.45, 0.75),
        x_drift_ratio=0.08,
    ):
        self.p = float(p)
        self.width_range = _parse_range(width_range, (0.2, 0.5))
        self.height_range = _parse_range(height_range, (0.2, 0.45))
        self.darkness_range = _parse_range(darkness_range, (0.45, 0.75))
        self.x_drift_ratio = float(x_drift_ratio)

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() >= self.p:
            return image, lanes, valid_mask, h_samples

        img_h, img_w = image.shape[:2]
        shadow_w = max(1, int(round(np.random.uniform(*self.width_range) * img_w)))
        shadow_h = max(1, int(round(np.random.uniform(*self.height_range) * img_h)))
        top_y = int(np.random.randint(0, max(img_h - shadow_h + 1, 1)))
        bottom_y = min(img_h - 1, top_y + shadow_h)

        top_x = int(np.random.randint(0, max(img_w - shadow_w + 1, 1)))
        drift = int(round(np.random.uniform(-self.x_drift_ratio, self.x_drift_ratio) * img_w))
        bottom_x = max(0, min(top_x + drift, img_w - shadow_w))

        polygon = np.array(
            [
                [top_x, top_y],
                [top_x + shadow_w, top_y],
                [bottom_x + shadow_w, bottom_y],
                [bottom_x, bottom_y],
            ],
            dtype=np.int32,
        )
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], color=1)

        darkness = float(np.random.uniform(*self.darkness_range))
        image = image.copy().astype(np.float32)
        image[mask > 0] *= darkness
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image, lanes, valid_mask, h_samples


class RandomBlur:
    def __init__(self, p=0.0, kernel_choices=(3, 5)):
        self.p = float(p)
        self.kernel_choices = tuple(int(k) for k in kernel_choices if int(k) % 2 == 1 and int(k) > 0)
        if not self.kernel_choices:
            raise ValueError("RandomBlur requires at least one positive odd kernel size.")

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() >= self.p:
            return image, lanes, valid_mask, h_samples

        kernel_size = int(np.random.choice(self.kernel_choices))
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
        return image, lanes, valid_mask, h_samples


class RandomTranslateShear:
    def __init__(
        self,
        p=0.0,
        max_translate_x_ratio=0.0,
        max_translate_y_ratio=0.0,
        max_shear_x_ratio=0.0,
    ):
        self.p = float(p)
        self.max_translate_x_ratio = float(max_translate_x_ratio)
        self.max_translate_y_ratio = float(max_translate_y_ratio)
        self.max_shear_x_ratio = float(max_shear_x_ratio)

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        if np.random.random() >= self.p:
            return image, lanes, valid_mask, h_samples

        img_h, img_w = image.shape[:2]
        tx = np.random.uniform(-self.max_translate_x_ratio, self.max_translate_x_ratio) * img_w
        ty = np.random.uniform(-self.max_translate_y_ratio, self.max_translate_y_ratio) * img_h
        shear_x = np.random.uniform(-self.max_shear_x_ratio, self.max_shear_x_ratio)

        matrix = build_translate_shear_matrix(img_h=img_h, tx=tx, ty=ty, shear_x=shear_x)
        image = cv2.warpAffine(
            image,
            matrix,
            (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        if lanes is not None and valid_mask is not None and h_samples is not None:
            lanes, valid_mask = warp_lane_rows(
                lanes=lanes,
                valid_mask=valid_mask,
                h_samples=h_samples,
                matrix=matrix,
                img_w=img_w,
                img_h=img_h,
            )

        return image, lanes, valid_mask, h_samples


class ResizeNormalize:
    def __init__(self, size, mean, std):
        self.size = tuple(size)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, lanes, valid_mask, h_samples=None):
        h, w = image.shape[:2]
        new_w, new_h = self.size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_h_samples = h_samples

        if lanes is not None and valid_mask is not None:
            x_scale = new_w / float(w)
            y_scale = new_h / float(h)
            lanes = lanes.copy()
            valid_mask = valid_mask.copy()

            # lanes: [N, num_y], x-values at fixed y-samples
            lanes = lanes * x_scale
            valid_mask = valid_mask.astype(np.uint8)
            if h_samples is not None:
                scaled_h_samples = np.round(np.array(h_samples, dtype=np.float32) * y_scale).astype(np.int32)

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        return image, lanes, valid_mask, scaled_h_samples


def build_transforms(cfg, is_train=False):
    size = cfg["dataset"]["img_size"]
    mean = cfg.get("dataset", {}).get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("dataset", {}).get("std", [0.229, 0.224, 0.225])
    
    if is_train:
        aug_cfg = cfg.get("augmentation", {})
        return Compose([
            RandomHorizontalFlip(p=aug_cfg.get("flip_p", 0.5)),
            RandomTranslateShear(
                p=aug_cfg.get("geom_p", 0.0),
                max_translate_x_ratio=aug_cfg.get("max_translate_x_ratio", 0.0),
                max_translate_y_ratio=aug_cfg.get("max_translate_y_ratio", 0.0),
                max_shear_x_ratio=aug_cfg.get("max_shear_x_ratio", 0.0),
            ),
            ColorJitter(
                brightness=aug_cfg.get("brightness", 0.3),
                contrast=aug_cfg.get("contrast", 0.3),
                saturation=aug_cfg.get("saturation", 0.3),
                p=aug_cfg.get("color_p", 0.8),
            ),
            RandomShadow(
                p=aug_cfg.get("shadow_p", 0.0),
                width_range=aug_cfg.get("shadow_width_range", [0.2, 0.5]),
                height_range=aug_cfg.get("shadow_height_range", [0.2, 0.45]),
                darkness_range=aug_cfg.get("shadow_darkness_range", [0.45, 0.75]),
                x_drift_ratio=aug_cfg.get("shadow_x_drift_ratio", 0.08),
            ),
            RandomOcclusion(
                p=aug_cfg.get("occlusion_p", 0.0),
                num_range=aug_cfg.get("occlusion_num_range", [1, 2]),
                width_range=aug_cfg.get("occlusion_width_range", [0.08, 0.18]),
                height_range=aug_cfg.get("occlusion_height_range", [0.08, 0.22]),
            ),
            RandomBlur(
                p=aug_cfg.get("blur_p", 0.0),
                kernel_choices=aug_cfg.get("blur_kernel_choices", [3, 5]),
            ),
            ResizeNormalize(size=size, mean=mean, std=std),
        ])
    else:
        return ResizeNormalize(size=size, mean=mean, std=std)
