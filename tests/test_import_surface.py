"""CI's lightweight contract: importing the agent must not load ML libraries.

CI deliberately installs no torch and no sentence-transformers so it finishes
in under a minute. That only holds while `agent.graph` keeps its heavy
dependencies lazy — `naive.rag` loads the embedder and cross-encoder inside
functions, not at module scope.

If someone adds a module-level `import torch`, CI does not merely slow down:
it fails at collection with ModuleNotFoundError, and a collection error stops
the whole suite. That is exactly how every test in this repository sat unrun
from the day CI was added until it was noticed on a pull request months later.
This test turns that silent, total failure into one obvious red test.
"""

import subprocess
import sys

HEAVY = ("torch", "sentence_transformers", "transformers")


def test_importing_the_agent_stays_lightweight():
    # A subprocess, because the local dev environment has these installed and
    # something else in this session may already have imported them.
    code = (
        "import sys; import agent.graph; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "", (
        f"agent.graph now imports {out} at module level. CI installs neither, "
        "so this would break collection and silently disable every test. "
        "Keep heavy imports inside functions (see naive.rag.get_embedder)."
    )
