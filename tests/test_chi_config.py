import dataclasses

import pytest

from modfish.chi.config import (
    ANTIALIAS_KINDS,
    FLAG_EMPTY,
    FLAG_MEANINGS,
    FLAG_NOISE,
    FLAG_RRHO,
    FLAG_SLOW,
    ChiParams,
)
from modfish.fctd.config import FCTDConfig


def test_defaults_match_spec():
    p = ChiParams()
    assert p.enabled is False
    assert p.gain is None
    assert p.antialias == "som_sinc4"
    assert p.noise_floor == 1e-9
    assert p.snr == 3.0
    assert (p.kmin, p.kmax_cap, p.fmax_cap) == (1.0, 12.5, 50.0)
    assert p.min_spd == 0.5
    assert (p.window, p.step, p.nsec) == (2.0, 0.25, 0.5)
    assert p.min_bins == 4
    assert p.closure is True
    assert p.closure_window == 31.25
    assert p.rrho_factor_max == 5.0
    assert (p.gamma, p.rho_0, p.g) == (0.2, 1026.0, 9.8)
    assert (p.nu, p.D, p.q) == (1.3e-6, 1.4e-7, 3.7)
    assert (p.R24, p.R25, p.R22, p.C19) == (1e6, 577e3, 266.1, 1e-6)
    assert p.gap == 0.01
    assert (p.rail_lo, p.rail_hi) == (0.001, 2.499)
    assert p.spd_smooth == 1.0


def test_flags_are_distinct_bits_and_documented():
    bits = [FLAG_SLOW, FLAG_NOISE, FLAG_EMPTY, FLAG_RRHO]
    assert bits == [1, 2, 4, 128]
    for word in ("slow", "noise_limited", "band_empty", "eps_table_edge", "n2_not_positive",
                 "rail", "no_closure_inputs", "rrho_capped"):
        assert word in FLAG_MEANINGS


def test_enabled_requires_gain():
    with pytest.raises(ValueError, match="gain"):
        ChiParams(enabled=True)


def test_antialias_kind_validated():
    assert ANTIALIAS_KINDS == ("som_sinc4", "ap00_sinc2")
    with pytest.raises(ValueError, match="antialias"):
        ChiParams(antialias="bessel")


def test_fctd_config_carries_chi_section():
    cfg = FCTDConfig.from_dict({"chi": {"enabled": True, "gain": 50.0, "gain_source": "d09 fit"}})
    assert cfg.chi.enabled is True
    assert cfg.chi.gain == 50.0
    assert FCTDConfig().chi == ChiParams()
    with pytest.raises(ValueError, match="Unknown keys in chi"):
        FCTDConfig.from_dict({"chi": {"gian": 50.0}})
    assert "chi" in {f.name for f in dataclasses.fields(FCTDConfig)}
