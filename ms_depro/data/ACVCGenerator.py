"""
See https://github.com/msohaildanish/DivAlign/blob/main/maskrcnn_benchmark/data/datasets/ACVCGenerator.py
"""

import numpy as np
from PIL import Image as PILImage
import torch
from scipy.stats import truncnorm
from imagecorruptions import corrupt, get_corruption_names


class ACVCGenerator:
    def __init__(self):
        """
        19 out of 22 corruptions used in this generator are taken from ImageNet-10.C:
            - https://github.com/bethgelab/imagecorruptions
        """
        # Preparation
        self.configuration()

    def configuration(self):
        self.shuffle_count = 1
        self.current_index = 0

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def get_severity(self):
        return np.random.randint(1, 6)

    def draw_cicle(self, shape, diamiter):
        """
        Input:
        shape    : tuple (height, width)
        diameter : scalar

        Output:
        np.array of shape  that says True within a circle with diamiter =  around center
        """
        assert len(shape) == 2
        TF = np.zeros(shape, dtype="bool")
        center = np.array(TF.shape) / 2.0

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
        return TF

    def filter_circle(self, TFcircle, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp[TFcircle] = fft_img_channel[TFcircle]
        return temp

    def inv_FFT_all_channel(self, fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return img_reco

    def high_pass_filter(self, x, severity):
        x = x.astype("float32") / 255.
        c = [.01, .02, .03, .04, .05][severity - 1]

        d = int(c * x.shape[0])
        TFcircle = self.draw_cicle(shape=x.shape[:2], diamiter=d)
        TFcircle = ~TFcircle

        fft_img = np.zeros_like(x, dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

        # For each channel, pass filter
        fft_img_filtered = []
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel = fft_img[:, :, ichannel]
            temp = self.filter_circle(TFcircle, fft_img_channel)
            fft_img_filtered.append(temp)
        fft_img_filtered = np.array(fft_img_filtered)
        fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
        x = np.clip(np.abs(self.inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def constant_amplitude(self, x, severity):
        """
        A visual corruption based on amplitude information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.05, .1, .15, .2, .25][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Amplitude replacement
        beta = 1.0 - c
        x_abs = np.ones_like(x_abs) * max(0, beta)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def phase_scaling(self, x, severity):
        """
        A visual corruption based on phase information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.1, .2, .3, .4, .5][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Phase scaling
        alpha = 1.0 - c
        x_pha = x_pha * max(0, alpha)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def apply_corruption(self, x, corruption_name, severity=None):
        if severity is None:
            severity = self.get_severity()

        custom_corruptions = {"high_pass_filter": self.high_pass_filter,
                              "constant_amplitude": self.constant_amplitude,
                              "phase_scaling": self.phase_scaling}

        if corruption_name in get_corruption_names('all'):
            x = corrupt(np.array(x), corruption_name=corruption_name, severity=severity)
            x = PILImage.fromarray(x)

        elif corruption_name in custom_corruptions:
            x = custom_corruptions[corruption_name](x, severity=severity)

        else:
            assert True, "%s is not a supported corruption!" % corruption_name
        return x

    def acvc(self, x):
        i = np.random.randint(5, 22)
        corruption_func = {
                           0: "defocus_blur",
                           1: "glass_blur",
                           2: "gaussian_blur",
                           3: "motion_blur",
                           4: "speckle_noise",
                           5: "shot_noise",
                           6: "impulse_noise",
                           7: "gaussian_noise",
                           8: "jpeg_compression",
                           9: "pixelate",
                           10: "elastic_transform",
                           11: "brightness",
                           12: "saturate",
                           13: "contrast",
                           14: "high_pass_filter",
                           15: "constant_amplitude",
                           16: "phase_scaling"
                           }
        return self.apply_corruption(x, corruption_func[i])

    def corruption(self, x, segmentation_mask=None):
        x_ = np.copy(x)
        x_ = self.acvc(x_)
        return x_