import gc
import io
import os
import platform
import shutil
import subprocess
from pathlib import Path
from time import time, sleep
from typing import List, Optional, Callable

import filetype
import gradio as gr
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import onnxruntime as ort
import psutil
import requests
import scipy as sp
import soundfile as sf
import torch
from PIL import Image
from TTS.api import TTS
from demucs import pretrained
from demucs.apply import apply_model
from demucs.states import load_model
from openvoice_cli import se_extractor
from openvoice_cli.api import ToneColorConverter
from pyannote.audio import Pipeline
from pydub import AudioSegment, silence
from scipy.io import wavfile
from torch import nn as nn
from tqdm import tqdm

tts = None
timer = None
ui_args = None
last_time = time()
vram_supported = False
print_debug = False
project_uuid = None

if torch.cuda.is_available():
    vram_supported = True

models = {
    "freevc24": "voice_conversion_models/multilingual/vctk/freevc24",
    "audiosep": "Audio-AGI/AudioSep"
}

device = torch.device('cpu') if platform.system() == 'Darwin' else torch.device('cuda:0')


# region Debugging/Performance


def printt(msg, reset: bool = False):
    global ui_args, last_time, timer, print_debug

    graph = None
    if timer is None:
        timer = PerfTimer(print_log=True)
    if reset:
        graph = timer.make_graph()
        timer.reset()
    timer.print_log = print_debug
    timer.record(msg)
    if graph:
        return graph


class PerfTimer:
    def __init__(self, print_log=False):
        self.start = time()
        self.records = {}
        self.total = 0
        self.base_category = ''
        self.print_log = print_log
        self.subcategory_level = 0
        self.ram_records = []
        self.vram_records = []
        self.time_points = []

    def get_ram_usage(self):
        return psutil.Process().memory_info().rss / (1024 ** 3)  # GB

    def get_vram_usage(self):
        if vram_supported:  # Ensure vram_supported is defined and correctly determines if VRAM usage can be checked
            torch.cuda.synchronize()  # Wait for all kernels in all streams on a CUDA device to complete
            info = torch.cuda.memory_stats()  # Get detailed CUDA memory stats
            used = info['allocated_bytes.all.peak']  # Get peak allocated bytes
            return used / (1024 ** 3)  # Convert bytes to GB
        return 0

    def elapsed(self):
        end = time()
        res = end - self.start
        self.start = end
        return res

    def add_time_to_record(self, category, amount):
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    def record(self, category, extra_time=0, disable_log=False):
        e = self.elapsed()
        ram_usage = self.get_ram_usage()
        vram_usage = self.get_vram_usage()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time
        self.time_points.append(self.total)
        self.ram_records.append(ram_usage)
        self.vram_records.append(vram_usage)

        if self.print_log and not disable_log:
            # Calculate the dynamic width for the category part
            category_width = int(80 * 0.75) - 2 * self.subcategory_level
            stats_width = 80 - category_width - 2 * self.subcategory_level

            # Prepare the format strings for both parts
            category_fmt = "{:" + str(category_width) + "}"
            stats_fmt = "{:>" + str(stats_width) + "}"

            # Format the category part and stats part separately
            category_part = category_fmt.format(f"{'  ' * self.subcategory_level}{category}:")
            stats_part = stats_fmt.format(
                f"done in {e + extra_time:.3f}s, RAM: {ram_usage:.2f}GB, VRAM: {vram_usage:.2f}GB")

            # Combine and print the full line
            print(category_part + stats_part)

    def make_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ram_records, label='RAM Usage (GB)', marker='o')
        if vram_supported:
            plt.plot(self.time_points, self.vram_records, label='VRAM Usage (GB)', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Usage (GB)')
        plt.title('Performance Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        image = Image.open(img_buf)
        img_buf.close()

        return image

    def summary(self):
        res = f"{self.total:.1f}s"

        additions = [(category, time_taken) for category, time_taken in self.records.items() if
                     time_taken >= 0.1 and '/' not in category]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def dump(self):
        return {'total': self.total, 'records': self.records, 'ram_usage': self.ram_records,
                'vram_usage': self.vram_records, 'time_points': self.time_points}

    def reset(self):
        self.__init__()


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# endregion

# region Music Separation

def sep_music(audio_path: str, project_path: str, return_all: bool = False, vox_only: bool = True) -> str:
    progress = gr.Progress()

    def update_sep_progress(percent: int, desc: str = "Separating audio"):
        progress(percent, desc=desc)

    use_cuda = torch.cuda.is_available()
    output_file = update_filename(audio_path, "separated", project_path)
    out_files = separate_music([audio_path], os.path.dirname(output_file), cpu=not use_cuda,
                               update_percent_func=update_sep_progress, only_vocals=vox_only)
    if not out_files:
        return ""
    if return_all:
        return out_files
    for file in out_files:
        if "vocal" in file:
            print(f"Found vocal file: {file}")
            return file
    return ""


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
            ), axis=1
        )

        mix_waves = []
        i = 0
        while i < n_sample + pad:
            waves = np.array(mix_p[:, i:i + model.chunk_size])
            mix_waves.append(waves)
            i += gen_size

        mix_waves_np = np.array(mix_waves)
        del mix_waves  # Free up the list memory
        mix_waves = torch.tensor(mix_waves_np, dtype=torch.float32).to(device)
        del mix_waves_np  # Free up the numpy array memory

        with torch.no_grad():
            stft_res = model.stft(mix_waves)
            res = infer_session.run(None, {'input': stft_res.cpu().numpy()})[0]
            del stft_res  # Free up tensor memory
            ten = torch.tensor(res, device=device)
            tar_waves = model.istft(ten)
            tar_waves = tar_waves.cpu()
            tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]
            del tar_waves  # Free up tensor memory

        sources.append(tar_signal)

    if device == 'cuda':
        torch.cuda.empty_cache()  # Clear unused memory from PyTorch

    # print('Time demix base: {:.2f} sec'.format(time() - start_time))
    return np.array(sources)


def demix_full(mix, device, chunk_size, models, infer_session, overlap=0.75):
    start_time = time()

    step = int(chunk_size * (1 - overlap))
    result = np.zeros((1, 2, mix.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mix.shape[-1]), dtype=np.float32)

    for i in range(0, mix.shape[-1], step):
        start = i
        end = min(i + chunk_size, mix.shape[-1])
        mix_part = mix[:, start:end]
        sources = demix_base(mix_part, device, models, infer_session)
        result[..., start:end] += sources
        divider[..., start:end] += 1
        del sources  # Free up the numpy array memory after each chunk is processed

    sources = result / divider
    del result, divider  # Free up numpy arrays

    # print('Final shape: {} Overall time: {:.2f}'.format(sources.shape, time() - start_time))
    return sources


class EnsembleDemucsMDXMusicSeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """

    def __init__(self, cpu: bool = False, single_onnx: bool = False, use_kim_model_1: bool = False,
                 overlap_large: float = 0.75, overlap_small: float = 0.5, chunk_size: int = 1000000):
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
        model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoint"))
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
        printt(f"Processing file {current_file_number + 1} of {total_files}")
        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        printt("Expand dims")
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
        printt("Torch from numpy")
        overlap_large = self.overlap_large
        overlap_small = self.overlap_small

        total_steps = 5 if only_vocals else 8 + len(self.models) * 2  # Adjust total steps based on conditions
        step_counter = 0  # Initialize step counter

        # Function to update the progress
        def update_progress(message="Processing"):
            nonlocal step_counter
            step_counter += 1
            if update_percent_func is not None:
                progress = min(step_counter / total_steps, total_steps / total_steps)
                update_percent_func(progress, message)

        # Get Demics vocal only
        model = self.model_vocals_only
        printt("LoadModel")
        shifts = 1
        overlap = overlap_large
        vocals_demucs = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        printt("Apply model 1")
        update_progress("Demucs vocal only")

        vocals_demucs += 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        printt("Apply model 2")
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
        printt("Demix full 1")
        vocals_mdxb1 = sources1
        printt("Copy sources")
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
            printt("Demix full 2")
            # it's instrumental so need to invert
            instrum_mdxb2 = sources2
            printt("Demix full 2 - instrum_mdxb2")
            vocals_mdxb2 = mixed_sound_array.T - instrum_mdxb2
            printt("Demix full 2 - focals_mdxb2")
            weights = np.array([12, 8, 3])
            printt("Demix full 2 - weights")
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_mdxb2.T + weights[
                2] * vocals_demucs.T) / weights.sum()
            printt("Demix full 2 - vocals")
        else:
            weights = np.array([6, 1])
            printt("Demix full 2 - weights")
            vocals = (weights[0] * vocals_mdxb1.T + weights[1] * vocals_demucs.T) / weights.sum()
            printt("Demix full 2 - vocals")

        update_progress("Demix full 2 done.")

        # vocals
        separated_music_arrays['vocals'] = vocals
        printt("Set vocals out")
        output_sample_rates['vocals'] = sample_rate
        printt("Set vocals out sr")
        if not only_vocals:
            # Generate instrumental
            instrum = mixed_sound_array - vocals
            printt("Generate instrumental")
            audio = np.expand_dims(instrum.T, axis=0)
            printt("Expand dims")
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
            printt("Torch from numpy")
            all_outs = []
            model_names = ["drums", "bass", "other", "vocals"]
            for i, model in enumerate(self.models):
                if i == 0:
                    overlap = overlap_small
                elif i > 0:
                    overlap = overlap_large
                out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                      + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
                printt(f"Processed instrument: {model_names[i]}")
                update_progress(f"Processing instrument: {model_names[i]}")

                if i == 2:
                    # ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
                    out[2] = out[2] + out[4] + out[5]
                    out = out[:4]

                out[0] = self.weights_drums[i] * out[0]
                out[1] = self.weights_bass[i] * out[1]
                out[2] = self.weights_other[i] * out[2]
                out[3] = self.weights_vocals[i] * out[3]
                printt(f"Weighted instrument: {model_names[i]}")
                all_outs.append(out)
            out = np.array(all_outs).sum(axis=0)
            printt("Summed all instruments")
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()
            printt("Normalized all instruments")
            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            printt("Subtract drums and bass")
            res = np.clip(res, -1, 1)
            printt("Clip res")
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate
            printt("Set other out")

            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            printt("Subtract vocals and bass")
            res = np.clip(res, -1, 1)
            printt("Clip res")
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate
            printt("Set drums out")
            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            printt("Subtract vocals and drums")
            res = np.clip(res, -1, 1)
            printt("Clip res")
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate
            printt("Set bass out")

            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']
            printt("Set bass, drums, other")
            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other
            printt("Set other, drums, bass")
        else:
            update_progress("Separating vox from other...")
            # Combine all non-vocal sounds into the "other" track
            other = mixed_sound_array - vocals
            printt("Subtract vocals")
            separated_music_arrays['other'] = other
            output_sample_rates['other'] = sample_rate
            printt("Set other out")

        update_progress("Separation complete...")

        printt("Cleanup...")
        if not only_vocals:
            del instrum
            free_mem()
            printt("Delete instrum")

        del audio
        free_mem()
        printt("Delete audio")

        del vocals
        free_mem()
        printt("Delete vocals")

        del mixed_sound_array
        free_mem()
        printt("Delete mixed_sound_array")

        del vocals_demucs
        free_mem()
        printt("Delete vocals_demucs")

        del sources1
        free_mem()
        printt("Delete sources1")

        if not self.single_onnx:
            del sources2
            free_mem()
            printt("Delete sources2")

        if not only_vocals:
            del instrum_mdxb2
            free_mem()
            printt("Delete instrum_mdxb2")

            del weights
            free_mem()
            printt("Delete weights")

            del vocals_mdxb1
            free_mem()
            printt("Delete vocals_mdxb1")
        else:
            del weights
            free_mem()
            printt("Delete weights")

            del other
            free_mem()

            del vocals_mdxb1
            del vocals_mdxb2
            del instrum_mdxb2
            del model
            free_mem()
            printt("Delete other")

        return separated_music_arrays, output_sample_rates

    def cleanup(self):
        # Explicitly delete tensors and models
        for model in self.models:
            model.cpu()  # Move model to CPU, which helps in releasing the GPU memory
            del model  # Delete the model to decrease reference count

        del self.models  # Finally, remove the list holding the models
        torch.cuda.empty_cache()  # Suggest to PyTorch to clean up unused memory
        printt("Delete models")
        if hasattr(self, 'model_vocals_only'):
            del self.model_vocals_only
            printt("Deleted model_vocals_only params.")
        if hasattr(self, 'infer_session1'):
            self.infer_session1.end_profiling()
            del self.infer_session1  # Assuming this is an ONNX session
            printt("Deleted infer_session1.")
        if hasattr(self, 'infer_session2') and not self.single_onnx:
            self.infer_session2.end_profiling()
            del self.infer_session2  # Assuming this is an ONNX session
            printt("Deleted infer_session2.")
        if hasattr(self, 'mdx_models1'):
            del self.mdx_models1
            printt("Deleted mdx_models1.")
        if hasattr(self, 'mdx_models2'):
            del self.mdx_models2
            printt("Deleted mdx_models2.")

        free_mem()
        printt("Cleanup complete.")


class EnsembleDemucsMDXMusicSeparationModelLowGPU:
    """
    Doesn't do any separation just passes the input back as output
    """

    def __init__(self, cpu: bool = False, single_onnx: bool = False, use_kim_model_1: bool = False,
                 overlap_large: float = 0.75, overlap_small: float = 0.5, chunk_size: int = None):
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
        model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoint"))
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
        free_mem()

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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        free_mem()

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
        free_mem()

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
        free_mem()

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
            printt('Use fast large GPU memory version of code')
            model = EnsembleDemucsMDXMusicSeparationModel(cpu, single_onnx, use_kim_model_1, overlap_large,
                                                          overlap_small, chunk_size)
        else:
            printt('Use low GPU memory version of code')
            model = EnsembleDemucsMDXMusicSeparationModelLowGPU(cpu, single_onnx, use_kim_model_1, overlap_large,
                                                                overlap_small, chunk_size)
        printt('MusicSep model initialized')
    outputs = []
    for i, input_audio in enumerate(input_audio):
        printt('Separating:')
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        printt("Input audio: {} Sample rate: {}".format(audio.shape, sr))
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
            printt('File created: {}'.format(output_path))

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

        # Conditionally add 'inst2' if not only_vocals
        if not only_vocals:
            del inst2
            printt('Delete inst2')

        del audio
        printt('Delete audio')

        del result
        printt('Delete result')

        del inst
        printt('Delete inst')

        del sr
        printt('Delete sr')

        del all_instrum
        printt('Delete all_instrum')

    if update_percent_func is not None:
        val = 100
        update_percent_func(int(val), f"Separated {len(input_audio)} files")
    printt('All files separated, deleting model.')
    # If the model is not None and has a cleanup() method, call it
    if model is not None and hasattr(model, 'cleanup'):
        printt('Model has cleanup method, using it.')
        model.cleanup()
        printt('Model cleanup done.')
    del model
    printt('Model deleted.')
    free_mem()
    printt('CUDA cache cleared (twice).')
    return outputs


# endregion

# region Cloning
def clone_voice_tts(target_file: str, source_speaker: str, project_path: str):
    if not is_audio(target_file):
        return ""
    global tts
    load_tts()
    if not isinstance(tts, TTS):
        return ""
    out_file = update_filename(target_file, "cloned-tts", project_path)
    chunks = chunk_audio(target_file, 60, project_path)
    if not chunks:
        return ""
    converted = []
    for chunk in chunks:
        print(f"Converting chunk: {chunk}")
        out_name = os.path.join(out_file, "chunks", "converted", os.path.basename(chunk))
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        try:
            tts.voice_conversion_to_file(source_wav=chunk, target_wav=source_speaker, file_path=out_name)
        except Exception as e:
            print(f"Failed to convert {chunk} with error: {e}")
            # Copy the original chunk to the converted directory
            out_name = os.path.join(out_file, "chunks", "converted", os.path.basename(chunk))
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            shutil.copy(chunk, out_name)
            continue
        converted.append(out_name)
    print("Joining audio")
    out_file = join_audio(converted, out_file)
    return out_file


def clone_voice_openvoice(target_speaker: str, source_speaker: str, project_path: str):
    if not is_audio(target_speaker):
        return ""
    out_file = update_filename(target_speaker, "cloned-openvoice", project_path)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(current_dir, 'checkpoint')
    ckpt_converter = os.path.join(checkpoints_dir, 'converter')

    if not os.path.exists(ckpt_converter):
        os.makedirs(ckpt_converter, exist_ok=True)
        download_checkpoint(ckpt_converter)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    printt("Loading OpenVoice converter")
    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
    printt("Loading SE extractor")
    source_se, _ = se_extractor.get_se(target_speaker, tone_color_converter, vad=True)
    printt("Extracting SE for target speaker")
    target_se, _ = se_extractor.get_se(source_speaker, tone_color_converter, vad=True)
    printt("Extracted SE for target speaker")
    # Ensure output directory exists and is writable
    output_dir = os.path.dirname(out_file)
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Run the tone color converter
    tone_color_converter.convert(
        audio_src_path=target_speaker,
        src_se=source_se,
        tgt_se=target_se,
        output_path=out_file,
    )
    printt(f"Cloned voice to {out_file}")
    del tone_color_converter
    del source_se
    del target_se
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    printt(f"Clone cleanup complete.")
    return out_file


# endregion

# region Model Management

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def download_checkpoint(dest_dir):
    # Define paths
    model_path = Path(dest_dir)

    # Define files and their corresponding URLs
    files_to_download = {
        "checkpoint.pth": f"https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter/checkpoint.pth?download=true",
        "config.json": f"https://huggingface.co/myshell-ai/OpenVoice/raw/main/checkpoints/converter/config.json",
    }

    # Check and create directories
    os.makedirs(model_path, exist_ok=True)

    # Download files if they don't exist
    for filename, url in files_to_download.items():
        destination = model_path / filename
        if not destination.exists():
            print(f"[OpenVoice Converter] Downloading {filename}...")
            download_file(url, destination)


def get_project_path(filename: str) -> str:
    basename_no_ext = os.path.splitext(os.path.basename(filename))[0]
    basename = basename_no_ext.replace(" ", "_")
    project_path = os.path.join(os.path.dirname(__file__), "outputs", basename)
    os.makedirs(project_path, exist_ok=True)
    return project_path


def load_tts():
    global tts, device
    if tts is None:
        tts = TTS(model_name=models["freevc24"]).to(device)


# endregion

# region File Management

# Module level variable
def set_project_uuid(uuid):
    global project_uuid
    project_uuid = uuid


def process_input(tgt_file, project_path):
    if is_video(tgt_file):
        return extract_audio(tgt_file, project_path)
    if tgt_file.endswith(".mp3"):
        wav_file = tgt_file.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(tgt_file)
        audio.export(wav_file, format="wav")
        return wav_file
    return tgt_file


def is_video(video_path: str) -> bool:
    return os.path.isfile(video_path) and filetype.is_video(video_path)


def is_audio(audio_path: str) -> bool:
    return os.path.isfile(audio_path) and filetype.is_audio(audio_path)


def update_filename(filename: str, process_name: str, project_path: str = None) -> str:
    global project_uuid
    if process_name in filename:
        return filename
    extension = os.path.splitext(filename)[1]  # This already includes the dot (e.g., ".wav")
    filename = os.path.splitext(os.path.basename(filename))[0]
    dirname = os.path.dirname(filename) if project_path is None else project_path
    os.makedirs(dirname, exist_ok=True)
    filename_parts = filename.split("_")
    process_names = ["separated", "replaced", "cleaned", "joined", "cloned-tts", "cloned-openvoice", project_uuid]
    # Separate the process names from the filename
    f_process_names = [f for f in process_names if f in filename_parts and f != project_uuid]
    # Get the original filename parts
    filename_parts = [f for f in filename_parts if f not in f_process_names and f != project_uuid]
    filename_parts.append(project_uuid)
    filename_parts += f_process_names
    filename_parts.append(process_name)
    base_filename = "_".join(filename_parts) + extension  # Directly use `extension` here
    return os.path.join(dirname, base_filename)


# endregion

# region Audio Management


def transcribe_audio(tgt_file, project_path):
    # model_path = "G-Root/speaker-diarization-optimized"
    model_path = "pyannote/speaker-diarization-3.1"
    # TODO: Save this somewhere reasonable
    hub_token_file = os.path.join(os.path.dirname(__file__), "hub_token.txt")
    hub_token = None
    if os.path.exists(hub_token_file):
        with open(hub_token_file, "r") as f:
            hub_token = f.read().strip()
            if hub_token == "PUT_YOUR_HF_HUB_TOKEN_HERE":
                print("Please replace the placeholder in hub_token.txt with your Hugging Face Hub token.")
                return [tgt_file]
    pipeline = Pipeline.from_pretrained(model_path, use_auth_token="")
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
    # run the pipeline on an audio file
    diarization = pipeline(tgt_file)
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((turn.start, turn.end))
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    speaker_files = []
    for speaker in speakers:
        speaker_file = os.path.join(project_path, f"speaker_{speaker}.txt")
        print(f"Speaker {speaker}")
        with open(speaker_file, "w") as f:
            for start, end in speakers[speaker]:
                f.write(f"start={start}s|stop={end}s\n")
        speaker_files.append(speaker_file)
    del pipeline
    free_mem()
    return speaker_files


def separate_speakers(tgt_file, speaker_times, project_path):
    original_audio = AudioSegment.from_file(tgt_file)
    output_files = []
    speaker_id = 0
    for speaker_file in speaker_times:
        intervals = []
        with open(os.path.join(project_path, speaker_file), 'r') as file:
            for line in file.readlines():
                parts = line.strip().split('|')
                start = float(parts[0].split('=')[1].rstrip('s')) * 1000  # Convert seconds to milliseconds
                end = float(parts[1].split('=')[1].rstrip('s')) * 1000
                intervals.append((start, end))

        speaker_audio = AudioSegment.silent(duration=len(original_audio))
        for start, end in intervals:
            speaker_audio = speaker_audio.overlay(original_audio[start:end], position=start)

        output_path = os.path.join(project_path, f"separated_speaker_{speaker_id}.wav")
        speaker_audio.export(output_path, format='wav')
        output_files.append(output_path)
        print(f"Generated separated audio for speaker {speaker_id} at {output_path}")
        speaker_id += 1
    return output_files


def merge_stems(stem_files, tgt_file, project_path: str):
    combined = None
    print(f"Combining stems to {tgt_file}")
    # Define the output file name
    output_file = update_filename(tgt_file, "joined", project_path)
    # Load and combine each stem
    for file in stem_files:
        stem = None
        load_attempt = 0
        load_failed = True
        while load_attempt < 3 and load_failed:
            try:
                print(f"Loading stem: {file}")
                stem = AudioSegment.from_file(file)
                load_failed = False
                print(f"Loaded stem: {file}")
            except Exception as e:
                print(f"Failed to load {file} with error: {e}")
                load_attempt += 1
                sleep(1)
        if stem is None:
            print(f"Failed to load {file}. Skipping...")
            continue
        if combined is None:
            combined = stem
        else:
            combined = combined.overlay(stem)
    output_extension = os.path.splitext(output_file)[1]
    # Export the combined audio to a new file
    combined.export(output_file, format=output_extension.lstrip('.'))
    return output_file


def clean_audio(audio_path: str, project_path: str):
    print(f"Cleaning {audio_path}")
    # Read the wav file
    rate, data = wavfile.read(audio_path)
    # Handle stereo files
    if data.ndim > 1 and data.shape[1] == 2:
        print("Stereo file detected. Processing channels separately.")
        # Process each channel with reduce_noise
        left_channel = nr.reduce_noise(y=data[:, 0], sr=rate, use_torch=True)
        right_channel = nr.reduce_noise(y=data[:, 1], sr=rate, use_torch=True)

        # Recombine the channels
        enhanced_speech = np.vstack((left_channel, right_channel)).T
    else:
        # Mono file or unexpected shape
        enhanced_speech = nr.reduce_noise(y=data, sr=rate, use_torch=True)

    out_file = update_filename(audio_path, "cleaned", project_path)
    # Write the cleaned audio
    wavfile.write(out_file, rate, enhanced_speech.astype(np.int16))
    return out_file


def chunk_audio(audio_file: str, chunk_len: int, project_path: str):
    if not os.path.exists(audio_file):
        return []

    out_dir = os.path.join(project_path, "chunks", "original")
    converted_dir = os.path.join(project_path, "chunks", "converted")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(converted_dir, exist_ok=True)

    # Ensure the output directory is empty
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    for f in os.listdir(converted_dir):
        os.remove(os.path.join(converted_dir, f))

    subprocess.run(f"ffmpeg -i {audio_file} -f segment -segment_time {chunk_len} -c copy {out_dir}/out%03d.wav",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")]


def join_audio(chunks: list[str], out_name: str):
    if not chunks:
        return ""
    print(f"Joining {len(chunks)} chunks to {out_name}")
    out_dir = os.path.dirname(out_name)
    # Temporary file to list all audio chunks
    list_file_path = os.path.join(out_dir, "chunk_list.txt")
    with open(list_file_path, 'w') as list_file:
        for chunk in chunks:
            list_file.write(f"file '{chunk}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {list_file_path} -c copy {out_name}",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Set file permissions to read/write for owner and read for others
    os.chmod(out_name, 0o644)
    os.remove(list_file_path)
    # Remove the chunks
    shutil.rmtree(os.path.join(out_dir, "chunks"), ignore_errors=True)
    return out_name


def replace_audio(video_file: str, audio_file: str):
    if not os.path.exists(video_file) or not os.path.exists(audio_file):
        return
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    out_file = update_filename(audio_file, "replaced", out_dir)
    video_extension = os.path.splitext(video_file)[1]
    audio_extension = os.path.splitext(audio_file)[1]
    out_file = out_file.replace(audio_extension, video_extension)
    subprocess.run(f"ffmpeg -y -i {video_file} -i {audio_file} -c:v copy -map 0:v:0 -map 1:a:0 {out_file}",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_file


def extract_audio(video_file: str, out_dir: str):
    if not os.path.exists(video_file):
        return ""
    file_without_extension = os.path.splitext(video_file)[0]
    audio_file = file_without_extension + ".wav"
    audio_file = os.path.join(out_dir, os.path.basename(audio_file))
    subprocess.run(f"ffmpeg -y -i {video_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_file

# endregion
