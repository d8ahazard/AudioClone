from typing import Dict, NoReturn, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase

from models.base import init_layer, init_bn


class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size, device='cuda'):
        super(FiLM, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.condition_size = condition_size
        self.modules, _ = self.create_film_modules(
            film_meta=film_meta,
            ancestor_names=[],
        )

    def create_film_modules(self, film_meta, ancestor_names):
        modules = {}
        for module_name, value in film_meta.items():
            if isinstance(value, int):
                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)
                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value,
                    unique_module_name=unique_module_name,
                )
            elif isinstance(value, dict):
                ancestor_names.append(module_name)
                modules[module_name], _ = self.create_film_modules(
                    film_meta=value,
                    ancestor_names=ancestor_names,
                )
            ancestor_names.pop()
        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):
        layer = nn.Linear(self.condition_size, num_features).to(self.device)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)
        return layer

    def forward(self, conditions):
        conditions = conditions.to(self.device)
        film_dict = self.calculate_film_data(
            conditions=conditions,
            modules=self.modules,
        )
        return film_dict

    def calculate_film_data(self, conditions, modules):
        film_data = {}
        for module_name, module in modules.items():
            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]
            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)
        return film_data


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        has_film,
        device="cuda"
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum).to(self.device)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum).to(self.device)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        ).to(self.device)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        ).to(self.device)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ).to(self.device)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)


    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        input_tensor = input_tensor.to(self.device)
        b1 = film_dict['beta1']
        b2 = film_dict['beta2']

        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1, negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x) + b2, negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


class EncoderBlockRes1B(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple, downsample: Tuple, momentum: float, has_film, device='cuda'):
        super(EncoderBlockRes1B, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conv_block1 = ConvBlockRes(in_channels, out_channels, kernel_size, momentum, has_film, device=self.device)
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple, upsample: Tuple, momentum: float, has_film, device='cuda'):
        super(DecoderBlockRes1B, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        ).to(self.device)

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum).to(self.device)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, kernel_size, momentum, has_film, device=self.device)
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum).to(self.device)
        self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        concat_tensor = concat_tensor.to(self.device)
        b1 = film_dict['beta1'].to(self.device)

        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1))
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, film_dict['conv_block2'])

        return x


class ResUNet30_Base(nn.Module):
    def __init__(self, input_channels, output_channels, device='cuda'):
        super(ResUNet30_Base, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        window_size = 2048
        hop_size = 320
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3

        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}
        self.gpu_devices = []
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        self.gpu_devices.append(self.stft)
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        self.gpu_devices.append(self.istft)
        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)
        self.gpu_devices.append(self.bn0)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.gpu_devices.append(self.pre_conv)

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block1)

        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block2)
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block3)
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block4)
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block5)
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.encoder_block6)
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.conv_block7a)
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block1)
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block2)
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block3)
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block4)
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block5)
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.gpu_devices.append(self.decoder_block6)

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.gpu_devices.append(self.after_conv)

        self.init_weights()
        try:
            self.to(self.device)
            for module in self.gpu_devices:
                module.to(self.device)
        except Exception as e:
            print(f'Failed to move model to device: {e}')

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

    def forward(self, mixtures, film_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'])  # (bs, 384, T / 32, F / 64)
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict

    def spectrogram(self, input, eps=0.):
        input = input.to(self.device)
        (real, imag) = self.stft(input)
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.):
        # If the input is a torch.cuda.FloatTensor, convert it to a torch.FloatTensor
        input = input.to(self.device)
        (real, imag) = self.stft(input)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-10):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def wav_to_spectrogram(self, input, eps=0.):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, channel, :], eps=eps))

        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.

        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel: channel + 1, :, :] * cos,
                                       spectrogram[:, channel: channel + 1, :, :] * sin, length))

        output = torch.stack(wav_list, dim=1)
        return output


def get_film_meta(module):

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            film_meta['beta2'] = module.bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta

    return film_meta


class ResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(ResUNet30, self).__init__()

        self.base = ResUNet30_Base(
            input_channels=input_channels,
            output_channels=output_channels,
        )

        self.film_meta = get_film_meta(
            module=self.base,
        )

        self.film = FiLM(
            film_meta=self.film_meta,
            condition_size=condition_size
        )

    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        output_dict = self.base(
            mixtures=mixtures,
            film_dict=film_dict,
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]

        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in,
                film_dict=film_dict,
            )['waveform']

            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in,
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np


