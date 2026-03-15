import cv2
import numpy as np


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
            ColorJitter(
                brightness=aug_cfg.get("brightness", 0.3),
                contrast=aug_cfg.get("contrast", 0.3),
                saturation=aug_cfg.get("saturation", 0.3),
                p=aug_cfg.get("color_p", 0.8),
            ),
            ResizeNormalize(size=size, mean=mean, std=std),
        ])
    else:
        return ResizeNormalize(size=size, mean=mean, std=std)
