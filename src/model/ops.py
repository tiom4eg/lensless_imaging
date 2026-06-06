import torch
import torch.fft as fft
from scipy.fft import next_fast_len


def get_full_shape(sensor_hw, pad_factor=2, fft_friendly=True):
    h, w = sensor_hw
    H = int(round(pad_factor * h))
    W = int(round(pad_factor * w))
    if fft_friendly:
        H = int(next_fast_len(H))
        W = int(next_fast_len(W))
    return H, W


def center_slices(full_hw, sensor_hw):
    H, W = full_hw
    h, w = sensor_hw
    top = (H - h) // 2
    left = (W - w) // 2
    return slice(top, top + h), slice(left, left + w)


def pad(x, full_hw):
    h, w = x.shape[-2:]
    sh, sw = center_slices(full_hw, (h, w))
    out = x.new_zeros(x.shape[:-2] + tuple(full_hw))
    out[..., sh, sw] = x
    return out


def crop(X, sensor_hw):
    sh, sw = center_slices(X.shape[-2:], sensor_hw)
    return X[..., sh, sw]


def psf_to_otf(psf, full_hw):
    psf_full = pad(psf, full_hw)
    psf_full = fft.ifftshift(psf_full, dim=(-2, -1))
    return fft.rfft2(psf_full, dim=(-2, -1))


def conv(x, otf):
    return fft.irfft2(fft.rfft2(x, dim=(-2, -1)) * otf, s=x.shape[-2:], dim=(-2, -1))


def conv_adjoint(y, otf):
    return fft.irfft2(fft.rfft2(y, dim=(-2, -1)) * otf.conj(), s=y.shape[-2:], dim=(-2, -1))


def otf_power(otf):
    return otf.real**2 + otf.imag**2


def grad(x):
    gy = torch.roll(x, shifts=-1, dims=-2) - x
    gx = torch.roll(x, shifts=-1, dims=-1) - x
    return torch.stack((gy, gx), dim=-3)


def grad_adjoint(g):
    gy = g[..., 0, :, :]
    gx = g[..., 1, :, :]
    dy = torch.roll(gy, shifts=1, dims=-2) - gy
    dx = torch.roll(gx, shifts=1, dims=-1) - gx
    return dy + dx


def grad_power(full_hw, device=None, dtype=torch.float32):
    delta = torch.zeros(full_hw, device=device, dtype=dtype)
    delta[0, 0] = 1.0
    impulse = grad_adjoint(grad(delta))
    return fft.rfft2(impulse, dim=(-2, -1)).real


def soft_threshold(z, thresh):
    return torch.sign(z) * torch.clamp(z.abs() - thresh, min=0.0)
