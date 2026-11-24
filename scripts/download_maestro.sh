#!/usr/bin/env bash
# Download the MAESTRO v3.0.0 MIDI-only archive (~56 MB) into data/raw/.
# License: CC BY-NC-SA 4.0 -- the data is never committed.
set -e
mkdir -p data/raw && cd data/raw
curl -L -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q maestro-v3.0.0-midi.zip
rm maestro-v3.0.0-midi.zip
