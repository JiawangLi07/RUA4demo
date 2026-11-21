"""Reachable wrist viewpoints and collision-checked joint connections from robot CAD."""

import itertools
from heapq import heappop, heappush

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc


class ViewBank:
    def __init__(self, ik, eye, neutral, park):
        samples = qmc.Halton(7, scramble=True, seed=36).random(7500)
        lo, hi = ik.bounds
        broad = lo + (hi - lo) * samples[:3200]
        around = np.clip(
            neutral
            + (samples[3200:] * 2 - 1) * [1.45, 1.12, 1.65, 1.72, 2.0, 1.6, 1.7],
            lo,
            hi,
        )
        candidates = np.vstack([neutral, park, broad, around])
        palm = np.stack([ik.fk(q) for q in candidates])
        select = (
            (palm[:, 0, 3] > 0.15)
            & (palm[:, 0, 3] < 1.03)
            & (palm[:, 2, 3] > 0.76)
            & (palm[:, 2, 3] < 1.28)
        )
        self.q, self.palm = candidates[select], palm[select]
        self.optical = self.palm @ eye
        self.tree = cKDTree(self.q)
        self.neighbors = self.tree.query(self.q, k=13)[1][:, 1:]

    def looks(self, point, current):
        delta = point - self.optical[:, :3, 3]
        cam = np.einsum("nij,ni->nj", self.optical[:, :3, :3], delta)
        z = cam[:, 2]
        off = np.linalg.norm(cam[:, :2], axis=1) / np.maximum(z, 0.001)
        eligible = (
            (z > 0.065)
            & (z < 0.50)
            & (off < 0.41)
            & (self.optical[:, 2, 3] > point[2] + 0.02)
        )
        ids = np.flatnonzero(eligible)
        cost = (
            off[ids]
            + 2 * abs(z[ids] - 0.23)
            + 0.065 * np.linalg.norm(self.q[ids] - current, axis=1)
        )
        return [(self.palm[i], self.q[i]) for i in ids[np.argsort(cost)[:12]]]

    def route(self, start, goal, free):
        # Lazy A*: validate the proposed route, exclude a failed edge and search again.
        # Spherical neighborhoods are in the station's seven joint coordinates.
        n = len(self.q)
        q = np.vstack([self.q, start, goal])
        first = set(map(int, self.tree.query(start, k=16)[1]))
        last = set(map(int, self.tree.query(goal, k=16)[1]))
        blocked = set()
        allowed = set()
        for _ in range(22):
            work = [(0.0, 0.0, n)]
            parent = {n: None}
            dist = {n: 0.0}
            while work:
                _, cost, i = heappop(work)
                if i == n + 1:
                    break
                if cost > dist[i]:
                    continue
                ns = first if i == n else self.neighbors[i]
                if i in last:
                    ns = [*ns, n + 1]
                for j in ns:
                    j = int(j)
                    edge = tuple(sorted((i, j)))
                    if edge in blocked:
                        continue
                    weight = np.linalg.norm(q[j] - q[i])
                    d = cost + weight
                    if d >= dist.get(j, np.inf):
                        continue
                    dist[j] = d
                    parent[j] = i
                    heappush(work, (d + np.linalg.norm(q[j] - goal), d, j))
            if n + 1 not in parent:
                return None
            path = []
            i = n + 1
            while i is not None:
                path.append(i)
                i = parent[i]
            path.reverse()
            samples = [start]
            for i, j in itertools.pairwise(path):
                edge = tuple(sorted((i, j)))
                steps = max(3, int(np.max(abs(q[i] - q[j])) / 0.13) + 2)
                part = np.linspace(q[i], q[j], steps)[1:]
                if edge not in allowed and not all(free(v) for v in part):
                    blocked.add(edge)
                    break
                allowed.add(edge)
                samples.extend(part)
            else:
                return samples
        return None
