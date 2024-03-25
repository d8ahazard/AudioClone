import os
import random
import shutil
from time import sleep
from typing import List
import argparse

import gradio as gr
from pydub import AudioSegment

from audio_clone import printt, get_project_path, is_video, is_audio, sep_music, merge_stems, \
    clean_audio, clone_voice_tts, clone_voice_openvoice, replace_audio, process_input

args = None


def process_outputs(src_file: str, out_file: str):
    if out_file is None or not os.path.exists(out_file):
        audio_clone.set_project_uuid(None)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)
    if is_video(src_file):
        out_file = replace_audio(src_file, out_file)
        audio_clone.set_project_uuid(None)
        return gr.update(visible=True, value=out_file), gr.update(visible=False, value=None)
    if is_audio(src_file):
        if src_file.endswith(".mp3"):
            # Convert out_file to mp3
            mp3_file = out_file.replace(".wav", ".mp3")
            audio = AudioSegment.from_wav(out_file)
            audio.export(mp3_file, format="mp3")
            # Delete the wav file
            os.remove(out_file)
            audio_clone.set_project_uuid(None)
            return gr.update(visible=False, value=None), gr.update(visible=True, value=mp3_file)
        audio_clone.set_project_uuid(None)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=out_file)
    audio_clone.set_project_uuid(None)
    return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


def process_separate(tgt_file: str):
    global project_uuid
    printt("Processing separate", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
    project_path = get_project_path(tgt_file)
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
    printt(f"Separated audio: {output}")
    step += 1
    progress(step / total_steps, desc="Separation complete.")
    printt(f"Separation complete: {output}")
    return process_outputs(src_file, output)


def process_clean(tgt_file):
    global project_uuid
    printt("Processing clean", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
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
    printt("Cleaning")
    output = clean_audio(tgt_file, project_path)
    printt(f"Cleaned audio: {output}")
    step += 1
    progress(step / total_steps, desc="Cleaning complete.")

    return process_outputs(src_file, output)


def process_all(tgt_file: str, src_file: str, clone_type: str, options: List[str]):
    printt("Processing all", True)
    project_uuid = ''.join(random.choices('0123456789abcdef', k=6))
    audio_clone.set_project_uuid(project_uuid)
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
        printt("Cleaning")
        tgt_file = clean_audio(tgt_file, project_path)
        printt(f"Cleaned audio: {tgt_file}")
        step += 1
    if "Separate" in options:
        progress(step / total_steps, desc="Separating audio")
        printt("Separating with music separator...")
        stems = sep_music(tgt_file, project_path, True)
        printt(f"Separated: {stems}")
        if len(stems):
            for file in stems:
                if "vocal" in file:
                    tgt_file = file
                    vox_file = file
                    break

        step += 1
    transcript_file = audio_clone.transcribe_audio(tgt_file, project_path)
    if clone_type == "TTS":
        progress(step / total_steps, desc="Cloning with TTS")
        printt(f"Cloning with TTS: {tgt_file} -> {src_file}")
        output = clone_voice_tts(tgt_file, src_file, project_path)
        printt(f"Cloned with TTS: {output}")
        step += 1
    else:
        progress(step / total_steps, desc="Cloning with OpenVoice")
        printt(f"Cloning with OpenVoice: {tgt_file} -> {src_file}")
        output = clone_voice_openvoice(tgt_file, src_file, project_path)
        printt(f"Cloned with OpenVoice: {output}")
        step += 1
    if stems is not None:
        replaced = []
        # Replace vox_file with the cloned audio
        for stem in stems:
            if stem == vox_file:
                replaced.append(output)
            else:
                replaced.append(stem)
        printt(f"Merging stems")
        output = merge_stems(replaced, output, project_path)
        printt(f"Merged stems: {output}")
    progress(step / total_steps, desc="Cloning complete.")
    printt(f"Cloning complete: {output}")
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
            clone_type_select = gr.Radio(label="Clone Type", choices=["OpenVoice", "TTS"], value="OpenVoice")
            clone_options = gr.CheckboxGroup(label="Clone Options", choices=["Separate", "Clean"],
                                             value=["Separate"])

    output_elements = [video_output, audio_output]

    separate_button.click(fn=process_separate, inputs=[tgt_speaker], outputs=output_elements)
    clean_button.click(fn=process_clean, inputs=[tgt_speaker], outputs=output_elements)
    submit_button.click(fn=process_all, inputs=[tgt_speaker, src_speaker, clone_type_select, clone_options], outputs=output_elements)

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
        print("onnxruntime not found. Please install onnxruntime-gpu to use the GPU.")
        exit(1)

    execution_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in execution_providers:
        print("CUDA execution provider not found. Please install onnxruntime-gpu to use the GPU.")
        exit(1)
    else:
        print("CUDA execution provider found.")
    # Delete everything in the temp directory
    if not args.debug:
        temp_path = os.path.join(os.path.dirname(__file__), "temp")
        for file in os.listdir(temp_path):
            file_path = os.path.join(temp_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    app.queue()
    app.launch()
