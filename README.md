
# AudioClone: Audio Enhancement and Voice Cloning Web Application

This web application offers a suite of advanced audio processing features, including noise reduction, audio separation, music extraction, and voice cloning. Utilizing state-of-the-art machine learning models and audio processing techniques, it provides an intuitive interface for enhancing audio files and cloning voices with remarkable accuracy.

## Features

- **Noise Reduction**: Clean up audio recordings by removing background noise, enhancing the clarity and quality of the sound.
- **Music Extraction**: Separate musical components from audio files, enabling users to extract instrumental or vocal tracks.
- **Voice Cloning**: Clone the voice from an audio sample and apply it to another voice or text-to-speech output, using either traditional TTS (Text-to-Speech) methods or advanced voice cloning techniques.

## Technologies Used

- **Gradio**: For building the web interface that allows users to interact with the application.
- **PyTorch**: As the primary machine learning framework for running the deep learning models.
- **Python Libraries**: Such as `scipy`, `numpy`, `pydub`, and `noisereduce` for audio file manipulation and processing.
- **TTS and Voice Conversion**: Leveraging models from the TTS library for text-to-speech and voice conversion functionalities.
- **FFmpeg**: For audio and video file processing, such as extracting audio from video files, splitting audio into chunks, and merging audio files.

## Getting Started

### Prerequisites

- Python 3.6 or later
- Pip for Python package installation

### Installation

1. Clone this repository or download the source code.
2. Navigate to the root directory of the application.
3. Create a virtual environment. For example, using `venv`:

```
bash
python -m venv venv
source venv/bin/activate
```

3. Install the required Python packages:

```
bash
python ./install.py
```

### Running the Application

Execute the main script to start the web application:

```
bash
python ./main.py
```

This command launches a local server, and the web interface can be accessed through a web browser at the URL provided in the terminal output (typically `http://127.0.0.1:7860`).

### Usage

- **Upload an Audio or Video File**: Start by uploading the target file using the "Target File" field. The application supports `.mp4` and `.wav` formats.
- **Audio Separation and Noise Reduction**: Select the "Extract Audio" or "Clean Audio" options and provide the necessary prompts or configurations.
- **Voice Cloning**: Choose the cloning method (TTS or OpenVoice), upload a source speaker audio file, and initiate the cloning process.
- **Review Outputs**: The processed audio or video file can be previewed and downloaded directly from the web interface.

## Development and Contributions

This application is open-source, and contributions are welcome. Whether it's a feature request, bug report, or a pull request, your input is valued and appreciated.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

