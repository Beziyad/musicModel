#!/usr/bin/env bash
# Listen to a MIDI file: MIDI -> WAV with fluidsynth (+ a SoundFont), then play it.
# Usage: scripts/play.sh outputs/before_untrained.mid
set -e
MIDI="$1"
SF2="data/soundfonts/GeneralUser-GS.sf2"
WAV="${MIDI%.*}.wav"
# -n: no MIDI input device, -i: not interactive, -F: write to file, -r: sample rate
fluidsynth -ni -F "$WAV" -r 44100 "$SF2" "$MIDI"
echo "Wrote $WAV — playing (Ctrl+C to stop)"
afplay "$WAV"
