The Dextra rock-scissors-paper robot perception pipeline in Python (pyaer + TensorFlow).

See also:

- [Dextra project](https://sensors.ini.ch/research/projects/dextra)
- [Dextra tendon-driven hand](https://sensorsini.github.io/dextra-robot-hand/)
- [Arduino firmware](https://github.com/SensorsINI/Dextra-robot-hand-firmware)
- [jAER](https://jaerproject.org)
- [ROSHAMBO17 dataset](https://docs.google.com/document/d/1rOltN_BaOTAMbP1chzFZxPjN24eTdbzuCrCM4S2o6qA/edit?tab=t.0)

The pretrained network is a 16-bit quantized CNN trained on that dataset.

## Requirements

- **Python 3.9** (not 3.10+). The saved CNN runs on **TensorFlow 2.5.2** / Keras 2.5. That TF series has no Python 3.10 wheels; do not try a newer TensorFlow.
- OS: tested on Ubuntu 18.04 and 22.04. Windows works for `consumer.py` (including jAER mmap). Live `producer.py` / pyaer needs libcaer (Linux or Intel macOS; on Windows use WSL2).
- CUDA is optional. CPU TensorFlow 2.5.2 is enough; missing `cudart` logs are expected without a GPU.
- Optional hardware: inivation DAVIS camera; Dextra hand Arduino on USB serial ([firmware](https://github.com/SensorsINI/Dextra-robot-hand-firmware)).

## Setup with uv (Python 3.9)

[uv](https://docs.astral.sh/uv/) installs a managed CPython 3.9 and a project venv.

Install uv if needed:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then in this repo:

```bash
uv python install 3.9
uv venv --python 3.9
```

Activate the venv:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows
.venv\Scripts\activate
```

Install pinned packages (TensorFlow 2.5.2 and `protobuf==3.20.3`):

```bash
uv pip install -r requirements.txt
```

Confirm:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.keras.__version__)"
```

You should see `2.5.2` and `2.5.0`. This repo includes `.python-version` set to `3.9` so later `uv venv` / `uv run` keep that interpreter.

### Robot camera (`producer.py`) — libcaer + pyaer

Needed only for the live DAVIS producer, not for `consumer.py --jaer-mmap`.

```bash
sudo apt-get install libcaer-dev   # or https://gitlab.com/inivation/dv/libcaer
uv pip install pyaer
```

See [pyaer](https://github.com/duguyue100/pyaer). On Windows, use WSL2 and [usbipd-win](https://github.com/dorssel/usbipd-win) (VS Code: [usbip-connect](https://marketplace.visualstudio.com/items?itemName=thecreativedodo.usbip-connect)).

### Conda alternative

```bash
conda create -n roshambo python=3.9
conda activate roshambo
pip install -r requirements.txt
```

## Running the robot

`python -m roshambo` starts `producer` and `consumer` (UDP pickle of 64×64 frames). You can also run those scripts separately.

1. Connect DAVIS and the hand Arduino over USB.
2. Note the Arduino serial device (`dmesg` on Linux). Default is `SERIAL_PORT` in `globals_and_utils.py`.
3. With the venv active:

```bash
python -m roshambo
```

`consumer.py` with no extra flags still listens for `producer.py` on UDP (port 12000) and uses the serial port. That is the museum / standalone robot path.

## [jAER](https://jaerproject.org) shared-memory input (hello world)

[jAER](https://jaerproject.org) can replace `producer.py` / pyaer. `SharedMemoryDVSFrameSender` writes 64×64 uint8 event-count frames to a memory-mapped file (plus optional localhost TCP). UDP pickle from `producer.py` remains the default.

1. In jAER, add/enable **SharedMemoryDVSFrameSender**. Leave defaults: 64×64, `dvsGrayScale=16`, `rectifyPolarities=true`, `normalizeFrame=false`, `showFrames=true`. Note **mmapPath** (Linux/macOS typically `/tmp/jaer_dvs_frames.mmap`; Windows `%TEMP%\jaer_dvs_frames.mmap`) and **controlPort** (14100).
2. Play a live camera or an AEDAT file (sample: [Davis346 Roshambo throws](https://drive.google.com/file/d/1hEI4HMODwAu6Pm9P4oDecePbfv--Lwbg/view?usp=drive_link) from [DAVIS24](https://sites.google.com/view/davis24-davis-sample-data/home); chip **Davis346blue**).
3. In this venv:

```bash
python consumer.py --jaer-mmap /tmp/jaer_dvs_frames.mmap --serial_port None --windowed
```

On Windows, pass the same path jAER shows for `mmapPath`. `--jaer-tcp 127.0.0.1:14100` is the default with `--jaer-mmap`; use `--jaer-tcp None` to poll mmap sequence numbers only. `--windowed` shows the CNN in a 640×640 window instead of fullscreen.

CNN weights stay in this project (`model/`); jAER does not need TensorFlow.

## Museum kiosk

For unattended operation, rtcwake must be allowed to suspend the machine across reboots.

- Copy [99-userdev-input.rules](99-userdev-input.rules) into `/etc/udev/rules.d`
- Copy [power-state.conf](power-state.conf) into `/etc/tmpfiles.d`
- Copy [dextra.desktop](dextra.desktop) and [symbols/dextra-icon.png](symbols/dextra-icon.png) to `~/.local/share/applications` and `~/.config/autostart`
- Enable autologin

Edit those files for your username.
