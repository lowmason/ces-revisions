"""The model-agnostic NUTS run harness: starts, sampling, diagnostics, and provenance.

``inits`` places dispersed chain starts from the caller's prior draws, ``fit`` runs NUTS
and writes a run directory, ``diagnostics`` checks the draws against Req 18's MCMC
thresholds, ``inference_data`` attaches the per-month and per-benchmark-year log
likelihood that blocked PSIS-LOO reads and stores the DataTree as netCDF, and
``provenance`` records the code, data, library, and host state of a run. Nothing here
knows a particular model: a model is a NumPyro function, its arguments, and a function
from a parameter dict to its per-step log likelihood.
"""
