"""Req 9 joint NSA/SA maps and release-specific B→M components."""

from functools import cache

import annual_data
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import vintage_data
from jax.experimental.sparse import BCOO

from ces_revisions.operators.post_march import build_recoverability
from ces_revisions.operators.seasonal import (
    build_b_to_m_components,
    implied_adjustment_operator,
    seasonal_measurement_operator,
)


@cache
def components() -> pl.DataFrame:
    recoverability = build_recoverability(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )
    return build_b_to_m_components(
        vintage_data.levels(), vintage_data.labels(), recoverability
    )


def test_seasonal_measurement_map_uses_one_state_not_a_second_free_path():
    operator = seasonal_measurement_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (2, 2)
    result = jax.jit(lambda z: operator @ z)(
        jnp.asarray([150.0, 12.0], dtype=jnp.float64)
    )
    np.testing.assert_array_equal(result, [150.0, 138.0])


def test_implied_adjustment_is_published_nsa_minus_sa():
    operator = implied_adjustment_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (1, 2)
    result = operator @ jnp.asarray([150.0, 138.0], dtype=jnp.float64)
    np.testing.assert_array_equal(result, [12.0])


def test_b_to_m_components_have_one_unique_row_per_axis_cell():
    frame = components()
    key = ["sector", "reference_month", "seasonal_status"]
    assert frame.select(key).n_unique() == frame.height
    assert set(frame["seasonal_status"]) == {"NSA", "SA"}
    assert set(frame["b_status"]) <= {
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    }
    assert set(frame["m_status"]) <= {
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    }


def test_sa_b_to_m_identity_uses_pairs_from_each_rows_own_release():
    observed = components().filter(
        (pl.col("seasonal_status") == "SA")
        & (pl.col("b_status") == "observed")
        & (pl.col("m_status") == "observed")
    )
    np.testing.assert_allclose(
        observed["observed_delta_thousands"],
        observed["paired_nsa_delta_thousands"]
        - observed["seasonal_adjustment_delta_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(observed["identity_residual_thousands"], 0.0)


def test_nsa_next_wedge_component_exists_only_for_april_through_october():
    frame = components()
    weighted = frame.filter(pl.col("next_wedge_weight").is_not_null())
    assert weighted["seasonal_status"].unique().to_list() == ["NSA"]
    assert weighted["reference_month"].dt.month().is_between(4, 10).all()
    expected = (weighted["reference_month"].dt.month() - 3) / 12
    np.testing.assert_allclose(weighted["next_wedge_weight"], expected)
    assert weighted["reconstruction_error_variance_thousands2"].gt(0.0).all()


def test_right_censored_m_rows_have_no_fabricated_value():
    rows = components().filter(pl.col("m_status") == "right_censored")
    assert rows.height > 0
    assert rows["m_value_thousands"].is_null().all()
    assert rows["observed_delta_thousands"].is_null().all()


def test_sa_and_nsa_m_horizons_remain_distinct():
    sample = components().filter(
        (pl.col("sector") == "00") & (pl.col("reference_month") == pl.date(2003, 5, 1))
    )
    release_by_status = dict(
        sample.select("seasonal_status", "m_release_month").iter_rows()
    )
    assert release_by_status["NSA"] != release_by_status["SA"]
