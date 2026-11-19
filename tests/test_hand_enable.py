import pytest
from exploration.hand_enable import await_enable_state


def test_delayed_and_transient_feedback():
    values = iter([False,False,True,False,True,True,True])
    observed = []
    await_enable_state(lambda:next(values),True,lambda:observed.append(1))
    assert len(observed)==7


def test_timeout_and_fault_do_not_confirm():
    with pytest.raises(RuntimeError,match="stabilize"):
        await_enable_state(lambda:False, True, lambda:None, timeout=.02)
    def fault():
        raise RuntimeError("drift")
    with pytest.raises(RuntimeError,match="drift"):
        await_enable_state(lambda:True, True, fault)
