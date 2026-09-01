import pytest

from modfish.fctd.config import FCTDConfig


def test_defaults_match_spec():
    cfg = FCTDConfig()
    assert cfg.casts.wlim == 0.4
    assert cfg.tc.N == 128
    assert cfg.tc.f0 == 6.0
    assert cfg.tc.thermal_mass is False
    assert cfg.grid.dz == 0.5
    assert cfg.latitude_fallback is None


def test_from_dict_partial_override():
    cfg = FCTDConfig.from_dict({"tc": {"N": 512}, "latitude_fallback": 2.0})
    assert cfg.tc.N == 512
    assert cfg.tc.f0 == 6.0
    assert cfg.latitude_fallback == 2.0


def test_from_dict_unknown_key_raises():
    with pytest.raises(ValueError, match="wlmi"):
        FCTDConfig.from_dict({"casts": {"wlmi": 0.3}})


def test_from_yaml_roundtrip(tmp_path):
    p = tmp_path / "fctd.yml"
    p.write_text("tc:\n  thermal_mass: true\ngrid:\n  dz: 1.0\n")
    cfg = FCTDConfig.from_yaml(p)
    assert cfg.tc.thermal_mass is True
    assert cfg.grid.dz == 1.0


def test_from_dict_malformed_nested_section_raises():
    with pytest.raises(ValueError, match="casts"):
        FCTDConfig.from_dict({"casts": 5})


def test_from_dict_nested_section_as_list_raises():
    with pytest.raises(ValueError, match="tc"):
        FCTDConfig.from_dict({"tc": ["smooth", 20]})
