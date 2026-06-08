from lensless_helpers.preprocessor import ALIGNMENT

ROI_TOP = ALIGNMENT["top_left"][0]
ROI_LEFT = ALIGNMENT["top_left"][1]
ROI_HEIGHT = ALIGNMENT["height"]
ROI_WIDTH = ALIGNMENT["width"]


def roi_crop(x):
    return x[..., ROI_TOP:ROI_TOP + ROI_HEIGHT, ROI_LEFT:ROI_LEFT + ROI_WIDTH]


def normalize_minmax(x, eps=1e-8):
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)
