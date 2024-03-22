import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from models.clap_encoder import CLAP_Encoder
from utils import ignore_warnings, parse_yaml, load_ss_model


def build_audiosep(config_yaml, checkpoint_path, device):
    ignore_warnings()
    configs = parse_yaml(config_yaml)

    query_encoder = CLAP_Encoder().eval()
    model = load_ss_model(configs=configs, checkpoint_path=checkpoint_path, query_encoder=query_encoder).eval().to(
        device)

    print(f'Loaded AudioSep model from [{checkpoint_path}]')
    return model


def separate_audio(model, audio_file, text, output_file, device='cuda', use_chunk=False):
    print(f'Separating audio from [{audio_file}] with textual query: [{text}]')
    mixture, fs = librosa.load(audio_file, sr=32000, mono=True)
    with torch.no_grad():
        text = [text]

        conditions = model.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }

        if use_chunk:
            sep_segment = model.ss_model.chunk_inference(input_dict)
            sep_segment = np.squeeze(sep_segment)
        else:
            sep_segment = model.ss_model(input_dict)["waveform"]
            sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

        write(output_file, 32000, np.round(sep_segment * 32767).astype(np.int16))
        print(f'Separated audio written to [{output_file}]')
