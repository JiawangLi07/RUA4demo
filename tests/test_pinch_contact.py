import pytest

from exploration.pinch_contact import FIRM_CONTACT_FLOOR, PinchSample, judge_pinch


def sample(frame, thumb=0.0, index=0.0, gap=.04, hand_sequence=None):
    return PinchSample((frame, frame), (frame / 30, frame / 30),
                       ((thumb, 0, 0), (0, index, 0)), gap,
                       frame if hand_sequence is None else hand_sequence)


BASELINE = [sample(frame) for frame in range(8)]


def test_live_pinch_requires_post_close_feedback():
    assert judge_pinch(BASELINE, [])["reason"] == "awaiting_post_close_feedback"


@pytest.mark.parametrize(("samples", "reason"), [
    ([sample(i, .08, .09, .024) for i in range(8, 11)], "bilateral_contact_confirmed"),
    ([sample(i, .08, .0, .024) for i in range(8, 11)], "bilateral_contact_not_sustained"),
    ([sample(8, .08, .09, .024)] * 3, "stale_tactile_frames"),
    ([sample(i, .08, .09, .024, hand_sequence=8) for i in range(8, 11)], "stale_hand_feedback"),
    ([sample(i, .08, .09, .04) for i in range(8, 11)], "bilateral_contact_confirmed"),
    ([sample(i, .08, .09, .024 + .002 * (i - 8)) for i in range(8, 11)], "bilateral_contact_confirmed"),
    ([sample(8, .08, .09, .024), sample(9, .08, .09, .024), sample(10, .08, 0, .024)],
     "bilateral_contact_not_sustained"),
])
def test_pinch_contact_decisions(samples, reason):
    result = judge_pinch(BASELINE, samples)
    assert result["reason"] == reason
    assert result["confirmed"] == (reason == "bilateral_contact_confirmed")


def test_contact_pattern_distinguishes_a_miss_from_an_anchored_finger():
    miss = judge_pinch(BASELINE, [sample(i, 0, 0, .024) for i in range(8, 11)])
    anchored = judge_pinch(BASELINE, [sample(i, .08, 0, .024) for i in range(8, 11)])
    assert miss["contact_pattern"] == "none" and miss["contacting_digits"] == []
    assert anchored["contact_pattern"] == "single" and anchored["contacting_digits"] == ["thumb"]


def test_light_threshold_is_fixed_even_with_noisy_baseline():
    loaded = [sample(i, .000538, .000364, .033) for i in range(8, 11)]
    idle = [sample(i, .000003, .000006, .033) for i in range(8, 11)]
    assert judge_pinch(BASELINE, loaded)["confirmed"]
    assert not judge_pinch(BASELINE, loaded, floor=FIRM_CONTACT_FLOOR)["confirmed"]

    noisy = [sample(i, (-1) ** i * .00012, (-1) ** i * .00002) for i in range(8)]
    assert judge_pinch(noisy, idle)["contact_pattern"] == "none"
    result = judge_pinch(noisy, loaded)
    assert result["contact_threshold_f6"] == [.00001, .00001]
    assert result["contacting_digits"] == ["thumb", "index"]


def test_light_contact_uses_the_requested_floor_when_baseline_is_quiet():
    light = [sample(i, .000012, .000013, .033) for i in range(8, 11)]
    result = judge_pinch(BASELINE, light)
    assert result["contact_threshold_f6"] == [.00001, .00001]
    assert result["confirmed"]


def test_startup_force_baseline_is_independent_of_later_preshape():
    baseline = [sample(i, gap=None) for i in range(8)]
    held = [sample(i, .0005, .0005, .029) for i in range(8, 11)]
    result = judge_pinch(baseline, held, precontact_gap_m=.034)
    assert result["confirmed"]
    assert result["pad_gap_reduction_m"] == pytest.approx(.005)
