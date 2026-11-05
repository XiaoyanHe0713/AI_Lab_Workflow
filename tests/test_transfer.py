"""Transfer blocks on every target and the settings of a target relative to a source set (instances, regime,
structure, the open flag recorded and never ranked); the replay provider's tool rounds."""
import json

import pytest

from lab.ledger import prompt_hash
from lab.roles import ReplayProvider, conversation_opening, strip_thinking
from lab.targets import SETTINGS, get_target, list_targets, transfer_setting


def test_every_target_has_a_transfer_block():
    for tid in list_targets():
        t = get_target(tid)
        b = t.transfer
        assert b is not None and b.instance == tid and b.certificate == t.certifier_name and b.family == t.family
        assert b.structure and b.regime == b.regime and b.content_terms and b.family_key == f"{b.family}/{b.structure}"
        assert b.open == (t.status == "open")
        assert t.class_definition


def test_quadrature_coordinates():
    b = get_target("quad-square-d5").transfer
    assert (b.family, b.structure, b.regime, b.open) == ("quadrature", "square", 5.0, False)
    assert get_target("quad-triangle-d6").transfer.open is True
    assert get_target("pep-davis-yin-contraction-calibration").transfer.regime == pytest.approx(100.0)   # L/mu = 10/0.1
    assert get_target("ssp-s3-p3").transfer.structure == "ssp" and get_target("stab-real-s4-p4").transfer.structure == "real_axis"
    assert get_target("opcount-dft-n16").transfer.regime == 16.0 and get_target("order-ode-p4").transfer.structure == "ode_ivp"


@pytest.mark.parametrize("sources,target,expected", [
    (["quad-d7"], "quad-d7", "same_instances"),
    (["quad-d7"], "quad-d7-pi", "new_instances"),
    (["quad-d7"], "quad-d9", "harder_regime"),
    (["quad-d9"], "quad-d7", "easier_regime"),
    (["quad-d7"], "quad-square-d5", "new_family_same_certificate"),
    (["quad-square-d5", "quad-d7"], "quad-triangle-d4", "new_family_same_certificate"),
    (["quad-d7"], "ssp-s3-p3", "new_domain"),
    (["ssp-s3-p3"], "stab-real-s4-p4", "new_family_same_certificate"),
    (["ssp-s3-p3"], "ssp-s5-p4", "harder_regime"),
    (["order-ode-p2"], "order-ode-p4", "harder_regime"),
])
def test_settings(sources, target, expected):
    st = transfer_setting(sources, target)
    assert st["setting"] == expected and expected in SETTINGS
    assert st["same_family"] == (expected in ("same_instances", "new_instances", "harder_regime", "easier_regime"))
    assert st["open"] == get_target(target).transfer.open and st["target_family_key"] == get_target(target).transfer.family_key


def test_open_flag_is_recorded_not_ranked():
    st = transfer_setting(["quad-triangle-d4"], "quad-triangle-d6")
    assert st["setting"] == "harder_regime" and st["open"] is True
    with pytest.raises(ValueError):
        transfer_setting([], "quad-d7")


def test_replay_rounds_by_conversation_opening(tmp_path):
    opening = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]
    key = prompt_hash(opening).split(":")[1][:24]
    (tmp_path / f"{key}.seq.json").write_text(json.dumps([{"text": "r1 seed0", "rounds": ["r1 seed0", "r2 seed0"]},
                                                           {"text": "r1 seed1", "rounds": ["r1 seed1", "r2 seed1"]}]))
    p = ReplayProvider(tmp_path, kind="local", model="m")
    assert p.complete(opening, seed=1).text == "r1 seed1"
    later = opening + [{"role": "assistant", "content": "r1 seed1"}, {"role": "user", "content": "tool result"}]
    assert conversation_opening(later) == opening
    assert p.complete(later, seed=1).text == "r2 seed1" and p.complete(later, seed=0).text == "r2 seed0"
    third = later + [{"role": "assistant", "content": "r2"}, {"role": "user", "content": "more"}]
    with pytest.raises(FileNotFoundError, match="round 3"):
        p.complete(third, seed=0)
    assert strip_thinking("<think>\nhmm\n</think>\n\nanswer") == "answer" and strip_thinking("plain") == "plain"
