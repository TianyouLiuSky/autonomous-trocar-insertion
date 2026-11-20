# Source from the package root in EVERY laboratory terminal.
: "${ISMR_SESSION:?Initialize and export ISMR_SESSION first}"
export C="data/sessions/$ISMR_SESSION/calibration"
export R="data/sessions/$ISMR_SESSION/raw_samples"
export L="data/sessions/$ISMR_SESSION/logs"
export T="data/sessions/$ISMR_SESSION/trials"
export PYTHONPATH="$PWD:$PWD/microscope_autonomy${PYTHONPATH:+:$PYTHONPATH}"
