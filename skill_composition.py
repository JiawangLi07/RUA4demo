"""Freeze verified Cells skills into an ordered, parameterized Python program."""
from hashlib import sha256
from textwrap import indent


def compose(entries):
    """The embedded sources pin revisions independently of later library changes."""
    components = []
    lines = ["def run_steps(robo, params):", "    steps = params['steps']"]
    for i, (entry, legacy_args) in enumerate(entries):
        source = entry["source"]
        if sha256(source.encode()).hexdigest() != entry["revision"]:
            raise ValueError("Component revision mismatch")
        components.append(f"# component {i}: {entry['revision']}\ndef component_{i}(robo, goal):\n"
                          + indent(source.strip(), "    ") + "\n    return run(robo, "
                          + (legacy_args or "goal") + ")\n")
        lines.append(f"    yield component_{i}(robo, steps[{i}])")
    return "\n".join(components + lines + ["", "def run(robo, params):",
                                           "    return list(run_steps(robo, params))", ""])
