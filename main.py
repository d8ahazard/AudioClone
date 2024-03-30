import argparse
import os
import random
import shutil
import subprocess
from typing import List

import gradio as gr
from pydub import AudioSegment

from audio_clone import printt, get_project_path, is_video, is_audio, sep_music, merge_stems, \
    clean_audio, clone_voice_tts, clone_voice_openvoice, replace_audio, process_input

args = None


def process_outputs(src_file: str, out_file: str, filename_parts: List[str] = None, restore_video: bool = True):
    printt("Processing outputs")
    out_folder = os.path.join(os.path.dirname(__file__), "outputs")
    if filename_parts is not None:
        base_filename = "_".join(filename_parts)
    else:
        base_filename = os.path.splitext(os.path.basename(src_file))[0]
    dest_file = os.path.join(out_folder, base_filename)
    if out_file is None or not os.path.exists(out_file):
        audio_clone.set_project_uuid(None)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)
    if is_video(src_file) and restore_video:
        out_file = replace_audio(src_file, out_file)
        printt(f"Replacing audio in video: {src_file} -> {out_file}")
        out_extension = os.path.splitext(src_file)[1]
        dest_file = dest_file + out_extension
        shutil.move(out_file, dest_file)
        printt(f"Replaced audio in video: {dest_file}")
        audio_clone.set_project_uuid(None)
        return gr.update(visible=True, value=dest_file), gr.update(visible=False, value=None)
    if is_audio(out_file) and not restore_video:
        src_file = out_file
    if is_audio(src_file):
        printt(f"Processing audio: {src_file} -> {out_file}")
        if src_file.endswith(".mp3"):
            # Convert out_file to mp3
            dest_file = dest_file + ".mp3"
            audio = AudioSegment.from_wav(out_file)
            audio.export(dest_file, format="mp3")
            # Delete the wav file
            os.remove(out_file)
        else:
            dest_file = dest_file + ".wav"
            shutil.move(out_file, dest_file)
        audio_clone.set_project_uuid(None)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=dest_file)

    audio_clone.set_project_uuid(None)
    return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


def process_separate(tgt_file: str, speaker_idx: int, sep_options: List[str]):
    global project_uuid
    printt("Processing separate", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
    project_path = get_project_path(tgt_file)
    base_filename = os.path.splitext(os.path.basename(tgt_file))[0]
    filename_parts = [base_filename, project_uuid, "separated"]
    src_file = tgt_file
    step = 0
    progress = gr.Progress()
    total_steps = 1
    if is_video(tgt_file):
        total_steps += 1
        progress(step / total_steps, desc="Extracting audio")
        step += 1
    printt(f"Processing inputs: {tgt_file}")
    tgt_file = process_input(tgt_file, project_path)
    printt(f"Processed inputs: {tgt_file}")
    progress(step / total_steps, desc="Separating audio with MusicSep")
    printt("Separating audio with musicSep")
    output = sep_music(tgt_file, project_path)
    printt(f"Separated audio")
    if "Transcribe" in sep_options:
        speaker_times = audio_clone.transcribe_audio(output, project_path)
        printt(f"Extracted speaker times")
        # TODO: Allow user to select speaker
        if len(speaker_times) >= speaker_idx + 1:
            printt(f"Multiple speakers detected, selecting {speaker_idx}")
            speakers = audio_clone.separate_speakers(tgt_file, speaker_times, project_path)
            # Store this so it can be replaced when combining stems
            output = speakers[speaker_idx]
    step += 1
    progress(step / total_steps, desc="Separation complete.")
    printt(f"Separation complete: {output}")
    return process_outputs(src_file, output, filename_parts, False)


def process_clean(tgt_file):
    global project_uuid
    printt("Processing clean", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
    project_path = get_project_path(tgt_file)
    base_filename = os.path.splitext(os.path.basename(tgt_file))[0]
    filename_parts = [base_filename, project_uuid, "cleaned"]
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
    printt("Cleaning")
    output = clean_audio(tgt_file, project_path)
    printt(f"Cleaned audio: {output}")
    step += 1
    progress(step / total_steps, desc="Cleaning complete.")

    return process_outputs(src_file, output, filename_parts, False)


def process_all(tgt_file: str, src_file: str, clone_type: str, options: List[str], speaker_idx: int, sep_options: List[str]):
    printt("Processing all", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
    project_path = get_project_path(tgt_file)
    base_filename = os.path.splitext(os.path.basename(tgt_file))[0]
    filename_parts = [base_filename, project_uuid]
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
    if "Clean" in options:
        progress(step / total_steps, desc="Cleaning audio")
        printt("Cleaning")
        tgt_file = clean_audio(tgt_file, project_path)
        printt(f"Cleaned audio: {tgt_file}")
        filename_parts.append("cleaned")
        step += 1
    if "Separate" in options:
        progress(step / total_steps, desc="Separating audio")
        printt("Separating with music separator...")
        stems = sep_music(tgt_file, project_path, True)
        printt(f"Separated: {stems}")
        filename_parts.append("separated")
        if len(stems):
            for file in stems:
                if "vocal" in file:
                    tgt_file = file
                    vox_file = file
                    break
        stems = [f for f in stems if f != tgt_file]
        step += 1
    if "Transcribe" in sep_options:
        speaker_times = audio_clone.transcribe_audio(tgt_file, project_path)
        if len(speaker_times) >= speaker_idx + 1:
            printt(f"Multiple speakers detected, selecting {speaker_idx}")
            speakers = audio_clone.separate_speakers(tgt_file, speaker_times, project_path)
            # Store this so it can be replaced when combining stems
            tgt_file = speakers[speaker_idx]
            other_files = [f for f in speakers if f != tgt_file]
            stems = stems + other_files
    if clone_type == "TTS":
        progress(step / total_steps, desc="Cloning with TTS")
        printt(f"Cloning with TTS: {tgt_file} -> {src_file}")
        output = clone_voice_tts(tgt_file, src_file, project_path)
        printt(f"Cloned with TTS: {output}")
        filename_parts.append("cloned-tts")
        step += 1
    else:
        progress(step / total_steps, desc="Cloning with OpenVoice")
        printt(f"Cloning with OpenVoice: {tgt_file} -> {src_file}")
        output = clone_voice_openvoice(tgt_file, src_file, project_path)
        printt(f"Cloned with OpenVoice: {output}")
        filename_parts.append("cloned-openvoice")
        step += 1
    if stems is not None:
        stems.append(output)
        printt(f"Merging stems")
        output = merge_stems(stems, output, project_path)
        printt(f"Merged stems: {output}")
    progress(step / total_steps, desc="Cloning complete.")
    printt(f"Cloning complete: {output}")
    return process_outputs(src_tgt, output, filename_parts, True)


def handle_tgt_speaker_change(tgt_path):
    # Assuming is_video_fn and is_audio_fn are functions that determine the file type
    target_video = gr.update(visible=False)
    target_audio = gr.update(visible=False)
    target_speaker = gr.update(visible=True)
    if tgt_path:
        print(f"File path: {tgt_path}")
        if is_video(tgt_path):
            target_video = gr.update(visible=True, value=tgt_path)
            target_audio = gr.update(visible=False)
            target_speaker = gr.update(value=tgt_path)
        elif is_audio(tgt_path):
            target_audio = gr.update(visible=True, value=tgt_path)
            target_video = gr.update(visible=False)
            target_speaker = gr.update(value=tgt_path)
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
favicon_path = os.path.join(os.path.dirname(__file__), "favicon.png")
with gr.Blocks(title="AudioClone", css=css_str) as app:
    with gr.Row():
        with gr.Column():
            gr.HTML("Inputs")
            with gr.Row():
                src_speaker = gr.Audio(label="Voice to Clone", type="filepath", elem_classes=["mediaItem"])
                tgt_speaker = gr.File(label="Destination File", type="filepath", file_types=[".mp4", ".wav"],
                                      elem_classes=["mediaItem"])
                tgt_video = gr.Video(label="Destination Video", visible=False, sources=["upload"],
                                     elem_classes=["mediaItem"], format="mp4")
                tgt_audio = gr.Audio(label="Destination Audio", visible=False, sources=["upload"],
                                     elem_classes=["mediaItem"])
        with gr.Column():
            gr.HTML("Outputs")
            with gr.Row():
                audio_output = gr.Audio(label="Output Audio", elem_classes=["mediaItem"])
                video_output = gr.Video(label="Output Video", visible=False, elem_classes=["mediaItem"], format="mp4")
    with gr.Row():
        with gr.Column():
            gr.HTML("Separating")
            with gr.Row():
                separate_options = gr.CheckboxGroup(label="Separate Options",
                                                    choices=["Transcribe", "Separate Instruments"],
                                                    value=["Transcribe"], interactive=True)
                target_speaker_select = gr.Dropdown(label="Speaker", choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], value=0)
                separate_button = gr.Button("Separate Voice")
    with gr.Row():
        with gr.Column():
            gr.HTML("Cleaning")
            with gr.Row():
                stationary_clean_radio = gr.Radio(label="Clean Type", choices=["Stationary", "Non-Stationary"],
                                                  value="Non-Stationary")
                noise_reduction_slider = gr.Slider(label="Noise Reduction", minimum=0, maximum=1, value=1, step=0.01)
                clean_button = gr.Button("Clean Voice")
    with gr.Row():
        with gr.Column():
            gr.HTML("Cloning")
            with gr.Row():
                clone_type_select = gr.Radio(label="Clone Type", choices=["OpenVoice", "TTS"], value="OpenVoice")
                clone_options = gr.CheckboxGroup(label="Clone Options", choices=["Separate", "Clean"],
                                                 value=["Separate"])
                submit_button = gr.Button("Clone Voice")

    output_elements = [video_output, audio_output]

    separate_button.click(fn=process_separate, inputs=[tgt_speaker, target_speaker_select, separate_options], outputs=output_elements)
    clean_button.click(fn=process_clean, inputs=[tgt_speaker], outputs=output_elements)
    submit_button.click(fn=process_all, inputs=[tgt_speaker, src_speaker, clone_type_select, clone_options, target_speaker_select, separate_options],
                        outputs=output_elements)

    tgt_speaker.upload(handle_tgt_speaker_change, inputs=[tgt_speaker], outputs=[tgt_video, tgt_audio, tgt_speaker])
    tgt_speaker.clear(handle_tgt_speaker_change, inputs=[tgt_speaker], outputs=[tgt_video, tgt_audio, tgt_speaker])

if __name__ == "__main__":
    # Create an argparser, and set a "debug" argument

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='127.0.0.1', help="IP address for the server to listen on.")
    parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
    parser.add_argument("--port", type=int, help="Port number for the server to listen on.")
    parser.add_argument("--outputs_folder", type=str, default='outputs',
                        help="Folder where output files will be saved.")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Enable debug mode, disables open_browser, and adds ui buttons for testing elements.")

    args = parser.parse_args()
    import audio_clone

    audio_clone.print_debug = args.debug
    # Ensure CUDA execution provider is aviailable for onnxruntime-gpu
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import onnxruntime
    except ImportError:
        print("Onnxruntime not found. Please run the install.py script and try again.")
        exit(1)

    try:
        import TTS
    except ImportError:
        print("TTS not found. Please run the install.py script and try again.")
        exit(1)

    execution_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in execution_providers:
        print("CUDA execution provider not found. Please install onnxruntime-gpu to use the GPU.")
        exit(1)
    else:
        print("CUDA execution provider found.")

    # Ensure FFMPEG is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("FFMPEG not found. Please install FFMPEG.")
        exit(1)

    hub_token_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hub_token.txt")
    if not os.path.exists(hub_token_file):
        with open(hub_token_file, "w") as f:
            f.write("Put your hub token here")

    # Delete everything in the temp directory
    temp_path = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_path, exist_ok=True)
    if not args.debug:
        for file in os.listdir(temp_path):
            file_path = os.path.join(temp_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Set the GRADIO_TEMP_DIR environment variable to the temp directory
    # os.environ["GRADIO_TEMP_DIR"] = temp_path
    # gradio_client.client.DEFAULT_TEMP_DIR = temp_path
    app.queue()
    app.launch(favicon_path=favicon_path)
