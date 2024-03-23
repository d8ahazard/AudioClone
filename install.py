import subprocess
import os
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
        print(f"Successfully {'un-' if uninstall else ''}installed {package}")


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
    pip_install(package)

torch_cmd = ["torch==2.1.0", "torchaudio==2.1.0", "torchlibrosa==0.1.0", "torchvision==0.16.0", "--index-url", "https://download.pytorch.org/whl/torch_stable.html"]
pip_install(torch_cmd)
pip_install(["onnxruntime-gpu", "onnxruntime"], True)
pip_install("onnxruntime-gpu")
