"""Measured contact closure; thresholds are supplied by the commissioned station."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ContactClosure:
    policy: dict
    baseline: np.ndarray

    def __post_init__(self):
        self.groups = self.policy["groups"]
        self.thresholds = self.policy["torque_delta"]
        self.required = int(self.policy["consecutive_samples"])
        if self.required < 2 or len(self.groups) < 2:
            raise ValueError("Contact requires two groups and consecutive samples")
        used = []
        for name, indices in self.groups.items():
            if not indices or any(type(i) is not int or not 0 <= i < 22 for i in indices):
                raise ValueError("Invalid contact channels")
            used.extend(indices)
            if not np.isfinite(self.thresholds[name]) or self.thresholds[name] <= 0:
                raise ValueError("Contact thresholds must be measured positive values")
        if len(set(used)) != len(used):
            raise ValueError("Contact groups overlap")
        self.count = dict.fromkeys(self.groups, 0)
        self.stopped = {}
        self.records = []

    def update(self, measured, torque, command):
        measured, torque, command = map(lambda v: np.asarray(v,dtype=float), (measured,torque,command))
        if any(v.shape != (22,) or not np.isfinite(v).all() for v in (measured,torque,command)):
            raise ValueError("Invalid contact feedback")
        for name, indices in self.groups.items():
            delta = float(max(abs(torque[indices]-self.baseline[indices])))
            self.count[name] = self.count[name]+1 if delta >= self.thresholds[name] else 0
            if self.count[name] >= self.required and name not in self.stopped:
                self.stopped[name] = measured[indices].copy()
                self.records.append({"group":name,"torque_delta":delta,"measured_rad":measured[indices].tolist()})
            if name in self.stopped:
                command[indices] = self.stopped[name]
        return command

    @property
    def confirmed(self):
        return len(self.stopped) == len(self.groups)
