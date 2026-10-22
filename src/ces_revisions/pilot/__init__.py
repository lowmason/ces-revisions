"""Roadmap Stage 6: the aggregate total-nonfarm linear-Gaussian pilot.

``panel`` defines the observation panel, ``assembly`` builds it from the Stage 3, 4, and 5
artifacts, ``layout`` and ``model`` define the state-space model and its priors,
``simulate`` draws synthetic panels from it, ``sbc`` checks its calibration, and
``dimensions`` sizes the Stage 7-9 models for the GPU measurement.
"""
