# coding: utf-8
__author__ = 'https://github.com/ZFTurbo/'

import gc
from typing import List, Optional, Callable

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import numpy as np
import torch
import torch.nn as nn
import os
import soundfile as sf

from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib

__VERSION__ = '1.0.1'


class Conv_TDF_net_trim_model(nn.Module):
    def __init__(self, device, target_name, L, n_fft, hop=1024):

        super(Conv_TDF_net_trim_model, self).__init__()

        self.dim_c = 4
        self.dim_f, self.dim_t = 3072, 256
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name

        out_c = self.dim_c * 4 if target_name == '*' else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

    def forward(self, x):
        x = self.first_conv(x)
        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.mid_dense(x)
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.us_dense[i](x)

        x = x.transpose(-1, -2)
        x = self.final_conv(x)
        return x


def get_models(name, device, load=True, vocals_model_type=0):
    model_vocals = None
    if vocals_model_type == 2:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=7680
        )
    elif vocals_model_type == 3:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=6144
        )

    return [model_vocals]


def demix_base(mix, device, models, infer_session):
    start_time = time()
    sources = []
    n_sample = mix.shape[1]
    for model in models:
        trim = model.n_fft // 2
        gen_size = model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        mix_p = np.concatenate(
            (
                np.zeros((2, trim)),
                mix,
                np.zeros((2, pad)),
                np.zeros((2, trim))
            ), 1
        )

        mix_waves = []
        i = 0
        mix_waves = []
        while i < n_sample + pad:
            waves = np.array(mix_p[:, i:i + model.chunk_size])
            mix_waves.append(waves)
            i += gen_size

        # Convert the list of NumPy arrays to a single NumPy array
        mix_waves_np = np.array(mix_waves)

        # Now convert the NumPy array to a PyTorch tensor
        mix_waves = torch.tensor(mix_waves_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            _ort = infer_session
            stft_res = model.stft(mix_waves)
            res = _ort.run(None, {'input': stft_res.cpu().numpy()})[0]
            ten = torch.tensor(res)
            tar_waves = model.istft(ten.to(device))
            tar_waves = tar_waves.cpu()
            tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

        sources.append(tar_signal)
    # print('Time demix base: {:.2f} sec'.format(time() - start_time))
    return np.array(sources)


def demix_full(mix, device, chunk_size, models, infer_session, overlap=0.75):
    start_time = time()

    step = int(chunk_size * (1 - overlap))
    # print('Initial shape: {} Chunk size: {} Step: {} Device: {}'.format(mix.shape, chunk_size, step, device))
    result = np.zeros((1, 2, mix.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mix.shape[-1]), dtype=np.float32)

    total = 0
    for i in range(0, mix.shape[-1], step):
        total += 1

        start = i
        end = min(i + chunk_size, mix.shape[-1])
        # print('Chunk: {} Start: {} End: {}'.format(total, start, end))
        mix_part = mix[:, start:end]
        sources = demix_base(mix_part, device, models, infer_session)
        # print(sources.shape)
        result[..., start:end] += sources
        divider[..., start:end] += 1
    sources = result / divider
    # print('Final shape: {} Overall time: {:.2f}'.format(sources.shape, time() - start_time))
    return sources


class EnsembleDemucsMDXMusicSeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self, cpu: bool = False, single_onnx: bool = False, use_kim_model_1: bool = False, overlap_large: float = 0.75, overlap_small: float = 0.5, chunk_size: int = 1000000):
        """
            options - user options
        """
        # print(options)

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if cpu:
            device = 'cpu'
        print('Use device: {}'.format(device))
        self.single_onnx = single_onnx
        if single_onnx:
            print('Use single vocal ONNX')

        self.kim_model_1 = use_kim_model_1
        if use_kim_model_1:
            print('Use Kim model 1')
        else:
            print('Use Kim model 2')

        self.overlap_large = overlap_large
        self.overlap_small = overlap_small
        if self.overlap_large > 0.99:
            self.overlap_large = 0.99
        if self.overlap_large < 0.0:
            self.overlap_large = 0.0
        if self.overlap_small > 0.99:
            self.overlap_small = 0.99
        if self.overlap_small < 0.0:
            self.overlap_small = 0.0
        model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "checkpoint")
        remote_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th'
        model_path = os.path.join(model_folder, '04573f0d-f3cf25b2.th')
        if not os.path.isfile(model_path):
            torch.hub.download_url_to_file(remote_url, model_path)
        model_vocals = load_model(model_path)
        model_vocals.to(device)
        self.model_vocals_only = model_vocals

        self.models = []
        self.weights_vocals = np.array([10, 1, 8, 9])
        self.weights_bass = np.array([19, 4, 5, 8])
        self.weights_drums = np.array([18, 2, 4, 9])
        self.weights_other = np.array([14, 2, 5, 10])

        model1 = pretrained.get_model('htdemucs_ft')
        model1.to(device)
        self.models.append(model1)

        model2 = pretrained.get_model('htdemucs')
        model2.to(device)
        self.models.append(model2)

        model3 = pretrained.get_model('htdemucs_6s')
        model3.to(device)
        self.models.append(model3)

        model4 = pretrained.get_model('hdemucs_mmi')
        model4.to(device)
        self.models.append(model4)

        # TODO: Use chunk_size from args
        if device == 'cpu':
            chunk_size = 200000000
            providers = ["CPUExecutionProvider"]
        else:
            chunk_size = 1000000
            providers = ["CUDAExecutionProvider"]

        # MDX-B model 1 initialization
        self.chunk_size = chunk_size
        self.mdx_models1 = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
        if self.kim_model_1:
            model_path_onnx1 = os.path.join(model_folder, 'Kim_Vocal_1.onnx')
            remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx'
        else:
            model_path_onnx1 = os.path.join(model_folder, 'Kim_Vocal_2.onnx')
            remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx'
        if not os.path.isfile(model_path_onnx1):
            torch.hub.download_url_to_file(remote_url_onnx1, model_path_onnx1)
        print('Model path: {}'.format(model_path_onnx1))
        print('Device: {} Chunk size: {}'.format(device, chunk_size))
        self.infer_session1 = ort.InferenceSession(
            model_path_onnx1,
            providers=providers,
            provider_options=[{"device_id": 0}],
        )

        if self.single_onnx is False:
            # MDX-B model 2  initialization
            self.chunk_size = chunk_size
            self.mdx_models2 = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
            root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
            model_path_onnx2 = os.path.join(model_folder, 'Kim_Inst.onnx')
            remote_url_onnx2 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Inst.onnx'
            if not os.path.isfile(model_path_onnx2):
                torch.hub.download_url_to_file(remote_url_onnx2, model_path_onnx2)
            print('Model path: {}'.format(model_path_onnx2))
            print('Device: {} Chunk size: {}'.format(device, chunk_size))
            self.infer_session2 = ort.InferenceSession(
                model_path_onnx2,
                providers=providers,
                provider_options=[{"device_id": 0}],
            )

        self.device = device
        pass

    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            update_percent_func=None,
            current_file_number=0,
            total_files=0,
            only_vocals=True,
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """


        # print('Update percent func: {}'.format(update_percent_func))

        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        overlap_large = self.overlap_large
        overlap_small = self.overlap_small

        total_steps = 5 if only_vocals else 8 + len(self.models) * 2  # Adjust total steps based on conditions
        step_counter = 0  # Initialize step counter

        # Function to update the progress
        def update_progress(message="Processing"):
            nonlocal step_counter
            step_counter += 1
            if update_percent_func is not None:
                progress = min(step_counter/total_steps, total_steps/total_steps)
                update_percent_func(progress, message)

        # Get Demics vocal only
        model = self.model_vocals_only
        shifts = 1
        overlap = overlap_large
        vocals_demucs = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()

        update_progress("Demucs vocal only")

        vocals_demucs += 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()

        update_progress("Demixing full..")

        overlap = overlap_large
        sources1 = demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models1,
            self.infer_session1,
            overlap=overlap
        )[0]

        vocals_mdxb1 = sources1

        update_progress("Demix full 1")

        if self.single_onnx is False:
            sources2 = -demix_full(
                -mixed_sound_array.T,
                self.device,
                self.chunk_size,
                self.mdx_models2,
                self.infer_session2,
                overlap=overlap
            )[0]

            # it's instrumental so need to invert
            instrum_mdxb2 = sources2
            vocals_mdxb2 = mixed_sound_array.T - instrum_mdxb2
            weights = np.array([12, 8, 3])
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_mdxb2.T + weights[
                2] * vocals_demucs.T) / weights.sum()
        else:
            weights = np.array([6, 1])
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_demucs.T) / weights.sum()

        update_progress("Demix full 2")

        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate

        if not only_vocals:
            # Generate instrumental
            instrum = mixed_sound_array - vocals

            audio = np.expand_dims(instrum.T, axis=0)
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

            all_outs = []
            model_names = ["drums", "bass", "other", "vocals"]
            for i, model in enumerate(self.models):
                if i == 0:
                    overlap = overlap_small
                elif i > 0:
                    overlap = overlap_large
                out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                      + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()

                update_progress(f"Processing instrument: {model_names[i]}")

                if i == 2:
                    # ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
                    out[2] = out[2] + out[4] + out[5]
                    out = out[:4]

                out[0] = self.weights_drums[i] * out[0]
                out[1] = self.weights_bass[i] * out[1]
                out[2] = self.weights_other[i] * out[2]
                out[3] = self.weights_vocals[i] * out[3]

                all_outs.append(out)
            out = np.array(all_outs).sum(axis=0)
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()

            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate

            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate

            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate

            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']

            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other
        else:
            update_progress("Separating vox from other...")
            # Combine all non-vocal sounds into the "other" track
            other = mixed_sound_array - vocals
            separated_music_arrays['other'] = other
            output_sample_rates['other'] = sample_rate

        update_progress("Separation complete...")

        return separated_music_arrays, output_sample_rates


class EnsembleDemucsMDXMusicSeparationModelLowGPU:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self, cpu: bool = False, single_onnx: bool = False, use_kim_model_1: bool = False, overlap_large: float = 0.75, overlap_small: float = 0.5, chunk_size: int = None):
        # print(options)

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if cpu:
            device = 'cpu'
        print('Use device: {}'.format(device))
        self.single_onnx = single_onnx
        self.kim_model_1 = use_kim_model_1
        if self.kim_model_1:
            print('Use Kim model 1')
        else:
            print('Use Kim model 2')

        self.overlap_large = overlap_large
        self.overlap_small = overlap_small
        if self.overlap_large > 0.99:
            self.overlap_large = 0.99
        if self.overlap_large < 0.0:
            self.overlap_large = 0.0
        if self.overlap_small > 0.99:
            self.overlap_small = 0.99
        if self.overlap_small < 0.0:
            self.overlap_small = 0.0

        self.weights_vocals = np.array([10, 1, 8, 9])
        self.weights_bass = np.array([19, 4, 5, 8])
        self.weights_drums = np.array([18, 2, 4, 9])
        self.weights_other = np.array([14, 2, 5, 10])

        # TODO: Use chunk_size from args
        if device == 'cpu':
            chunk_size = 200000000
            self.providers = ["CPUExecutionProvider"]
        else:
            chunk_size = 1000000
            self.providers = ["CUDAExecutionProvider"]
        self.chunk_size = chunk_size
        self.device = device
        pass

    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            update_percent_func=None,
            current_file_number=0,
            total_files=0,
            only_vocals=False
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # print('Update percent func: {}'.format(update_percent_func))

        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        overlap_large = self.overlap_large
        overlap_small = self.overlap_small

        # Get Demucs vocal only
        model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "checkpoint")
        remote_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th'
        model_path = os.path.join(model_folder, '04573f0d-f3cf25b2.th')
        os.makedirs(model_folder, exist_ok=True)
        if not os.path.isfile(model_path):
            torch.hub.download_url_to_file(remote_url, model_path)
        model_vocals = load_model(model_path)
        model_vocals.to(self.device)
        shifts = 1
        overlap = overlap_large
        vocals_demucs = 0.5 * apply_model(model_vocals, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.10) / total_files
            update_percent_func(int(val))

        vocals_demucs += 0.5 * -apply_model(model_vocals, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        del model_vocals
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.20) / total_files
            update_percent_func(int(val))

        # MDX-B model 1 initialization
        mdx_models1 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=2)
        if self.kim_model_1:
            model_path_onnx1 = os.path.join(model_folder, 'Kim_Vocal_1.onnx')
            remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx'
        else:
            model_path_onnx1 = os.path.join(model_folder, 'Kim_Vocal_2.onnx')
            remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx'
        if not os.path.isfile(model_path_onnx1):
            torch.hub.download_url_to_file(remote_url_onnx1, model_path_onnx1)
        print('Model path: {}'.format(model_path_onnx1))
        print('Device: {} Chunk size: {}'.format(self.device, self.chunk_size))
        infer_session1 = ort.InferenceSession(
            model_path_onnx1,
            providers=self.providers,
            provider_options=[{"device_id": 0}],
        )
        overlap = overlap_large
        sources1 = demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            mdx_models1,
            infer_session1,
            overlap=overlap
        )[0]
        vocals_mdxb1 = sources1
        del infer_session1
        del mdx_models1

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.30) / total_files
            update_percent_func(int(val))

        if self.single_onnx is False:
            # MDX-B model 2  initialization
            mdx_models2 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=2)
            root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
            model_path_onnx2 = os.path.join(model_folder, 'Kim_Inst.onnx')
            remote_url_onnx2 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Inst.onnx'
            if not os.path.isfile(model_path_onnx2):
                torch.hub.download_url_to_file(remote_url_onnx2, model_path_onnx2)
            print('Model path: {}'.format(model_path_onnx2))
            print('Device: {} Chunk size: {}'.format(self.device, self.chunk_size))
            infer_session2 = ort.InferenceSession(
                model_path_onnx2,
                providers=self.providers,
                provider_options=[{"device_id": 0}],
            )

            overlap = overlap_large
            sources2 = -demix_full(
                -mixed_sound_array.T,
                self.device,
                self.chunk_size,
                mdx_models2,
                infer_session2,
                overlap=overlap
            )[0]

            # it's instrumental so need to invert
            instrum_mdxb2 = sources2
            vocals_mdxb2 = mixed_sound_array.T - instrum_mdxb2
            del infer_session2
            del mdx_models2
            weights = np.array([12, 8, 3])
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_mdxb2.T + weights[
                2] * vocals_demucs.T) / weights.sum()
        else:
            weights = np.array([6, 1])
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_demucs.T) / weights.sum()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.40) / total_files
            update_percent_func(int(val))

        # Generate instrumental
        instrum = mixed_sound_array - vocals

        audio = np.expand_dims(instrum.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        all_outs = []

        i = 0
        overlap = overlap_small
        model = pretrained.get_model('htdemucs_ft')
        model.to(self.device)
        out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
              + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.50 + i * 0.10) / total_files
            update_percent_func(int(val))

        out[0] = self.weights_drums[i] * out[0]
        out[1] = self.weights_bass[i] * out[1]
        out[2] = self.weights_other[i] * out[2]
        out[3] = self.weights_vocals[i] * out[3]
        all_outs.append(out)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        i = 1
        overlap = overlap_large
        model = pretrained.get_model('htdemucs')
        model.to(self.device)
        out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
              + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.50 + i * 0.10) / total_files
            update_percent_func(int(val))

        out[0] = self.weights_drums[i] * out[0]
        out[1] = self.weights_bass[i] * out[1]
        out[2] = self.weights_other[i] * out[2]
        out[3] = self.weights_vocals[i] * out[3]
        all_outs.append(out)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        i = 2
        overlap = overlap_large
        model = pretrained.get_model('htdemucs_6s')
        model.to(self.device)
        out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
              + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.50 + i * 0.10) / total_files
            update_percent_func(int(val))

        # More stems need to add
        out[2] = out[2] + out[4] + out[5]
        out = out[:4]
        out[0] = self.weights_drums[i] * out[0]
        out[1] = self.weights_bass[i] * out[1]
        out[2] = self.weights_other[i] * out[2]
        out[3] = self.weights_vocals[i] * out[3]
        all_outs.append(out)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        i = 3
        model = pretrained.get_model('hdemucs_mmi')
        model.to(self.device)
        out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
              + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.50 + i * 0.10) / total_files
            update_percent_func(int(val))

        out[0] = self.weights_drums[i] * out[0]
        out[1] = self.weights_bass[i] * out[1]
        out[2] = self.weights_other[i] * out[2]
        out[3] = self.weights_vocals[i] * out[3]
        all_outs.append(out)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out = np.array(all_outs).sum(axis=0)
        out[0] = out[0] / self.weights_drums.sum()
        out[1] = out[1] / self.weights_bass.sum()
        out[2] = out[2] / self.weights_other.sum()
        out[3] = out[3] / self.weights_vocals.sum()

        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate

        # other
        res = mixed_sound_array - vocals - out[0].T - out[1].T
        res = np.clip(res, -1, 1)
        separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
        output_sample_rates['other'] = sample_rate

        # drums
        res = mixed_sound_array - vocals - out[1].T - out[2].T
        res = np.clip(res, -1, 1)
        separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
        output_sample_rates['drums'] = sample_rate

        # bass
        res = mixed_sound_array - vocals - out[0].T - out[2].T
        res = np.clip(res, -1, 1)
        separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
        output_sample_rates['bass'] = sample_rate

        bass = separated_music_arrays['bass']
        drums = separated_music_arrays['drums']
        other = separated_music_arrays['other']

        separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
        separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
        separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other

        if update_percent_func is not None:
            val = 100 * (current_file_number + 0.95) / total_files
            update_percent_func(int(val))

        return separated_music_arrays, output_sample_rates


def separate_music(input_audio: List[str], output_folder: str, cpu: bool = False, overlap_large: float = 0.6,
                   overlap_small: float = 0.5, single_onnx: bool = False, chunk_size: int = 1000000,
                   large_gpu: bool = False, use_kim_model_1: bool = False, only_vocals: bool = True,
                   update_percent_func: Optional[Callable[[int, str], None]] = None) -> List[str]:
    for file in input_audio:
        if not os.path.isfile(file):
            print('Error. No such file: {}. Please check path!'.format(file))
            return []

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # If CUDA is available, get the total VRAM
    total_vram = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_vram += torch.cuda.get_device_properties(i).total_memory
    # If total VRAM is greater than 16GB, use the large GPU memory version of the code
    if total_vram > 16 * 1024 * 1024 * 1024:
        large_gpu = True
    model = None
    if model is None:
        if large_gpu is True:
            print('Use fast large GPU memory version of code')
            model = EnsembleDemucsMDXMusicSeparationModel(cpu, single_onnx, use_kim_model_1, overlap_large, overlap_small, chunk_size)
        else:
            print('Use low GPU memory version of code')
            model = EnsembleDemucsMDXMusicSeparationModelLowGPU(cpu, single_onnx, use_kim_model_1, overlap_large, overlap_small, chunk_size)
    outputs = []
    for i, input_audio in enumerate(input_audio):
        print('Separating: {}'.format(input_audio))
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        print("Input audio: {} Sample rate: {}".format(audio.shape, sr))
        result, sample_rates = model.separate_music_file(
            audio.T,
            sr,
            update_percent_func,
            i,
            len(input_audio),
            only_vocals,
        )
        all_instrum = model.instruments
        if only_vocals:
            all_instrum = ['vocals']
        for instrum in all_instrum:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format(instrum)
            output_path = os.path.join(output_folder, output_name)
            sf.write(output_path, result[instrum], sample_rates[instrum], subtype='FLOAT')
            outputs.append(output_path)
            print('File created: {}'.format(output_path))

        # instrumental part 1
        inst = audio.T - result['vocals']
        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_instrum.wav'
        output_path = os.path.join(output_folder, output_name)
        sf.write(output_path, inst, sr, subtype='FLOAT')
        outputs.append(output_path)
        print('File created: {}'.format(output_path))

        if not only_vocals:
            # instrumental part 2
            inst2 = result['bass'] + result['drums'] + result['other']
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_instrum2.wav'
            output_path = os.path.join(output_folder, output_name)
            sf.write(output_path, inst2, sr, subtype='FLOAT')
            outputs.append(output_path)
            print('File created: {}'.format(output_path))

    if update_percent_func is not None:
        val = 100
        update_percent_func(int(val), f"Separated {len(input_audio)} files")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


