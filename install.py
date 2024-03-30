import subprocess
import os
import sys
from typing import Union, List


def pip_install(package: Union[str, List[str]], uninstall=False):
    if isinstance(package, str):
        command = ["pip", "install", package]
        if uninstall:
            command = ["pip", "uninstall", package, "-y"]
    else:
        command = ["pip", "install"] + package
        if uninstall:
            command = ["pip", "uninstall"] + package + ["-y"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to {'un-' if uninstall else ''}install {package}. Error: {result.stderr}")
    else:
        if "Successfully installed" in result.stdout:
            print(f"Successfully {'un-' if uninstall else ''}installed {package}")
        elif "already satisfied" in result.stdout:
            print(f"{package} is already installed")


def in_venv():
    return sys.prefix != sys.base_prefix


def create_venv():
    venv_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        result = subprocess.run(["python", "-m", "venv", venv_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to create virtual environment. Error: {result.stderr}")
            exit(1)
        else:
            print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")


if __name__ == "__main__":
    # Ensure we have a virtual environment
    create_venv()
    if not in_venv():
        print("Please run this script in a virtual environment.")
        exit(1)
    requirements_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    extra_indices = []
    packages = ["openvoice_cli"]
    if not os.path.exists(requirements_file):
        print("Cannot find requirements file.")
        exit(1)

    with open(requirements_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("--extra-index-url"):
                extra_indices.append(line)
            elif line and line not in packages:
                packages.append(line)

    for package in packages:
        print(f"Installing {package}")
        pip_install(package)

    torch_cmd = ["torch==2.1.0", "torchaudio==2.1.0", "torchlibrosa==0.1.0", "torchvision==0.16.0", "--index-url",
                 "https://download.pytorch.org/whl/cu121"]
    pip_install(torch_cmd)
    # Create the hub_token.txt file and write "Put your hub token here" in it.
    hub_token_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hub_token.txt")
    if not os.path.exists(hub_token_file):
        with open(hub_token_file, "w") as f:
            f.write("Put your hub token here")

    pip_install(["onnxruntime-gpu", "onnxruntime"], True)
    pip_install("onnxruntime-gpu")
