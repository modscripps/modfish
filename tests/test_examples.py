from pathlib import Path

from modfish.fctd import FCTDConfig

EXAMPLE = Path(__file__).parents[1] / "examples" / "fctd_example.yml"


def test_example_config_loads():
    cfg = FCTDConfig.from_yaml(EXAMPLE)
    assert cfg.chi.rail_lo == 0.05
    assert cfg.chi.rail_hi == 2.45
    assert cfg.chi.min_bins == 3


def test_example_config_tc_is_noop():
    # MOTIVE values are commented reference only
    cfg = FCTDConfig.from_yaml(EXAMPLE)
    assert cfg.tc.lag == 0 and cfg.tc.tau_t == 0
    assert not cfg.tc.thermal_mass
