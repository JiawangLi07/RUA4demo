import numpy as np
import pytest
from exploration.contact import ContactClosure
from exploration.hand_session import validate_session_cell


def test_contact_requires_consecutive_opposing_groups_and_freezes_each():
    c=ContactClosure({"groups":{"thumb":[0],"index":[5]},"torque_delta":{"thumb":1,"index":1},
                      "consecutive_samples":2},np.zeros(22))
    q=np.zeros(22)
    t=np.zeros(22);t[0]=2
    c.update(q,t,np.ones(22));assert not c.stopped
    command=c.update(q,t,np.ones(22));assert command[0]==0 and command[5]==1 and not c.confirmed
    t[5]=2
    c.update(q,t,np.ones(22));assert not c.confirmed
    c.update(q,t,np.ones(22));assert c.confirmed


def test_invalid_contact_policy_and_nonfinite_reading():
    with pytest.raises(ValueError):
        ContactClosure({"groups":{"a":[0]},"torque_delta":{"a":1},"consecutive_samples":2},np.zeros(22))
    c=ContactClosure({"groups":{"a":[0],"b":[1]},"torque_delta":{"a":1,"b":1},
                     "consecutive_samples":2},np.zeros(22))
    with pytest.raises(ValueError):c.update(np.full(22,np.nan),np.zeros(22),np.zeros(22))


def test_grasp_cell_is_wired():
    validate_session_cell("robo.grasp({'offset_m':[0,0,0], 'lift_m':0.065, 'grasp':'pinch', 'recheck':True})",[])
