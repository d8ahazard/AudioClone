import gc
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from time import sleep
from typing import List

import filetype
import gradio as gr
import noisereduce as nr
import numpy as np
import requests
import torch
from TTS.api import TTS
from openvoice_cli import se_extractor
from openvoice_cli.api import ToneColorConverter
from pydub import AudioSegment
from scipy.io import wavfile
from tqdm import tqdm

from models.audiosep import AudioSep
from music_sep.music_sep import separate_music
from pipeline import separate_audio
from utils import get_ss_model


def is_mac_os():
    return platform.system() == 'Darwin'


models = {
    "freevc24": "voice_conversion_models/multilingual/vctk/freevc24",
    "audiosep": "Audio-AGI/AudioSep"
}

device = torch.device('cpu') if is_mac_os() else torch.device('cuda:0')

tts = None
project_uuid = None


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


def is_video(video_path: str) -> bool:
    return os.path.isfile(video_path) and filetype.is_video(video_path)


def is_audio(audio_path: str) -> bool:
    return os.path.isfile(audio_path) and filetype.is_audio(audio_path)


def update_filename(filename: str, process_name: str, project_path: str = None) -> str:
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


def sep_audio(audio_path: str, sep_description: str, project_path: str) -> str:
    sep_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ss_model = get_ss_model('config/audiosep_base.yaml', sep_device)
    model = AudioSep.from_pretrained("nielsr/audiosep-demo", ss_model=ss_model)
    output_file = update_filename(audio_path, "separated", project_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    separate_audio(model, audio_path, sep_description, output_file, sep_device, use_chunk=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output_file


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

    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))

    source_se, _ = se_extractor.get_se(target_speaker, tone_color_converter, vad=True)
    target_se, _ = se_extractor.get_se(source_speaker, tone_color_converter, vad=True)

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
    out_file = update_filename(video_file, "replaced", out_dir)
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


def update_inputs(target_speaker):
    if is_video(target_speaker):
        return gr.update(visible=True), gr.update(visible=False)
    elif is_audio(target_speaker):
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


def process_input(tgt_file, project_path):
    if is_video(tgt_file):
        return extract_audio(tgt_file, project_path)
    if tgt_file.endswith(".mp3"):
        wav_file = tgt_file.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(tgt_file)
        audio.export(wav_file, format="wav")
        return wav_file
    return tgt_file


def process_outputs(src_file: str, out_file: str):
    if out_file is None or not os.path.exists(out_file):
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)
    if is_video(src_file):
        out_file = replace_audio(src_file, out_file)
        return gr.update(visible=True, value=out_file), gr.update(visible=False, value=None)
    if is_audio(src_file):
        if src_file.endswith(".mp3"):
            # Convert out_file to mp3
            mp3_file = out_file.replace(".wav", ".mp3")
            audio = AudioSegment.from_wav(out_file)
            audio.export(mp3_file, format="mp3")
            # Delete the wav file
            os.remove(out_file)
            return gr.update(visible=False, value=None), gr.update(visible=True, value=mp3_file)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=out_file)
    return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


def process_separate(tgt_file: str, sep_audio_prompt: str, sep_type):
    global project_uuid
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    project_path = get_project_path(tgt_file)
    src_file = tgt_file
    step = 0
    progress = gr.Progress()
    total_steps = 1
    if is_video(tgt_file):
        total_steps += 1
        progress(step / total_steps, desc="Extracting audio")
        step += 1
    tgt_file = process_input(tgt_file, project_path)

    if sep_type == "Music":
        progress(step / total_steps, desc="Separating music")
        output = sep_music(tgt_file, project_path)
    else:
        progress(step / total_steps, desc="Separating audio")
        output = sep_audio(tgt_file, sep_audio_prompt, project_path)
    step += 1
    progress(step / total_steps, desc="Separation complete.")
    return process_outputs(src_file, output)


def process_clean(tgt_file):
    global project_uuid
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    project_path = get_project_path(tgt_file)
    src_file = tgt_file
    step = 0
    progress = gr.Progress()
    total_steps = 1
    if is_video(tgt_file):
        total_steps += 1
        progress(step / total_steps, desc="Extracting audio")
        step += 1
    tgt_file = process_input(tgt_file, project_path)

    progress(step / total_steps, desc="Cleaning audio")
    output = clean_audio(tgt_file, project_path)
    step += 1
    progress(step / total_steps, desc="Cleaning complete.")

    return process_outputs(src_file, output)


def process_all(tgt_file: str, src_file: str, clone_type: str, options: List[str], sep_audio_prompt: str,
                sep_type: str):
    global project_uuid
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    project_path = get_project_path(tgt_file)
    src_tgt = tgt_file
    step = 0
    progress = gr.Progress()
    total_steps = 1 + len(options)
    if is_video(tgt_file):
        total_steps += 1
        progress(step / total_steps, desc="Extracting audio")
        step += 1
    tgt_file = process_input(tgt_file, project_path)
    stems = None
    vox_file = None
    if "Clean" in options:
        progress(step / total_steps, desc="Cleaning audio")
        print("Cleaning")
        tgt_file = clean_audio(tgt_file, project_path)
        step += 1
    if "Separate" in options:
        progress(step / total_steps, desc="Separating audio")
        if sep_type == "Music":
            print("Separating with music separator...")
            stems = sep_music(tgt_file, project_path, True)
            if len(stems):
                for file in stems:
                    if "vocal" in file:
                        tgt_file = file
                        vox_file = file
                        break
        else:
            print("Separating with audiosep...")
            tgt_file = sep_audio(tgt_file, sep_audio_prompt, project_path)
        step += 1
    if clone_type == "TTS":
        progress(step / total_steps, desc="Cloning with TTS")
        print(f"Cloning with TTS: {tgt_file} -> {src_file}")
        output = clone_voice_tts(tgt_file, src_file, project_path)
        step += 1
    else:
        progress(step / total_steps, desc="Cloning with OpenVoice")
        print(f"Cloning with OpenVoice: {tgt_file} -> {src_file}")
        output = clone_voice_openvoice(tgt_file, src_file, project_path)
        step += 1
    if stems is not None:
        replaced = []
        # Replace vox_file with the cloned audio
        for stem in stems:
            if stem == vox_file:
                replaced.append(output)
            else:
                replaced.append(stem)
        output = merge_stems(replaced, output, project_path)
    progress(step / total_steps, desc="Cloning complete.")
    return process_outputs(src_tgt, output)


def handle_tgt_speaker_change(file_path):
    # Assuming is_video_fn and is_audio_fn are functions that determine the file type
    target_video = gr.update(visible=False)
    target_audio = gr.update(visible=False)
    target_speaker = gr.update(visible=True)
    if file_path:
        print(f"File path: {file_path}")
        temp_path = os.path.join(os.path.dirname(__file__), "temp")
        # Move the file to the temp directory
        os.makedirs(temp_path, exist_ok=True)
        temp_file_path = os.path.join(temp_path, os.path.basename(file_path))
        if not os.path.exists(temp_file_path):
            shutil.move(file_path, temp_path)
            sleep(0.5)
        if os.path.exists(file_path):
            # Really delete it
            os.remove(file_path)
        file_path = temp_file_path
        if is_video(file_path):
            target_video = gr.update(visible=True, value=file_path)
            target_audio = gr.update(visible=False)
            target_speaker = gr.update(value=file_path)
        elif is_audio(file_path):
            target_audio = gr.update(visible=True, value=file_path)
            target_video = gr.update(visible=False)
            target_speaker = gr.update(value=file_path)
    else:
        # Show target_speaker if no file or an unsupported file type is uploaded
        target_speaker = gr.update()
        target_video = gr.update(visible=False)
        target_audio = gr.update(visible=False)
    return target_video, target_audio, target_speaker


def reset_components():
    # Function to reset the visibility of components when the audio or video is cleared
    target_speaker = gr.update(visible=True)
    target_video = gr.update(visible=False)
    target_audio = gr.update(visible=False)
    return target_video, target_audio, target_speaker


css_str = """
.mediaItem {
    max-height: 400px;
    height: 400px;
}
"""

with gr.Blocks(css=css_str) as app:
    with gr.Row():
        separate_button = gr.Button("Separate")
        clean_button = gr.Button("Clean")
        submit_button = gr.Button("Clone Voice")

    with gr.Column():
        with gr.Row():
            src_speaker = gr.Audio(label="Voice to Clone", type="filepath", elem_classes=["mediaItem"])
            tgt_speaker = gr.File(label="Destination File", type="filepath", file_types=[".mp4", ".wav"],
                                  elem_classes=["mediaItem"])
            tgt_video = gr.Video(label="Destination Video", visible=False, sources=["upload"],
                                 elem_classes=["mediaItem"])
            tgt_audio = gr.Audio(label="Destination Audio", visible=False, sources=["upload"],
                                 elem_classes=["mediaItem"])
            audio_output = gr.Audio(label="Output Audio", elem_classes=["mediaItem"])
            video_output = gr.Video(label="Output Video", visible=False, elem_classes=["mediaItem"])

    with gr.Column():
        with gr.Row():
            separation_type_select = gr.Radio(label="Separation Type", choices=["Music", "AudioSep"], value="Music")
            separation_audio_prompt = gr.Textbox(label="Separation Prompt", placeholder="A woman talking",
                                                 visible=False)
            clone_type_select = gr.Radio(label="Clone Type", choices=["OpenVoice", "TTS"], value="OpenVoice")
            clone_options = gr.CheckboxGroup(label="Clone Options", choices=["Separate", "Clean"],
                                             value=["Separate"])

    output_elements = [video_output, audio_output]

    separate_button.click(fn=process_separate, inputs=[tgt_speaker, separation_audio_prompt, separation_type_select],
                          outputs=output_elements)
    clean_button.click(fn=process_clean, inputs=[tgt_speaker], outputs=output_elements)
    submit_button.click(fn=process_all,
                        inputs=[tgt_speaker, src_speaker, clone_type_select, clone_options, separation_audio_prompt,
                                separation_type_select], outputs=output_elements)


    def update_sep_inputs(sep_type):
        return gr.update(visible=sep_type == "AudioSep")


    separation_type_select.change(fn=update_sep_inputs, inputs=[separation_type_select],
                                  outputs=[separation_audio_prompt])

    tgt_speaker.upload(handle_tgt_speaker_change, inputs=[tgt_speaker], outputs=[tgt_video, tgt_audio, tgt_speaker])
    tgt_speaker.clear(handle_tgt_speaker_change, inputs=[tgt_speaker], outputs=[tgt_video, tgt_audio, tgt_speaker])

if __name__ == "__main__":
    # Ensure CUDA execution provider is aviailable for onnxruntime-gpu
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import onnxruntime
    except ImportError:
        print("onnxruntime not found. Please install onnxruntime-gpu to use the GPU.")
        exit(1)

    execution_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in execution_providers:
        print("CUDA execution provider not found. Please install onnxruntime-gpu to use the GPU.")
        exit(1)
    else:
        print("CUDA execution provider found.")
    app.queue()
    app.launch()
