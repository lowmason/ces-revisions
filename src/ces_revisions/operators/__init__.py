"""Sparse deterministic operators shared by CES data preparation and models."""

from ces_revisions.vintages.raw import SUPERSECTORS

SECTOR_ORDER = tuple(SUPERSECTORS)
OBSERVATION_SECTOR_ORDER = ("00", *SECTOR_ORDER)

__all__ = ["OBSERVATION_SECTOR_ORDER", "SECTOR_ORDER"]
