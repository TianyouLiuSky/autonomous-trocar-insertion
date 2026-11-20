#!/bin/bash
# For CLI tools: change to package root and preserve its imports.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/microscope_autonomy${PYTHONPATH:+:$PYTHONPATH}"
: "${ISMR_SESSION:?Initialize and export ISMR_SESSION}"
exec python3 "$@"
