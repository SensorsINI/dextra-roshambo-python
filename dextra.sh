#!/usr/bin/bash -l
eval "$(conda shell.bash hook)"
conda activate dextra
cd /home/dextra/Dropbox/GitHub/SensorsINI/dextra-roshambo-python
python -m roshambo
