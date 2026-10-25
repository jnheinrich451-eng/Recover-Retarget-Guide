"""DiTFlow motion guidance ported to the Wan2.1 backbone.

STATUS: Phase-2 PROBE MATERIAL, not integrated. Landed to satisfy CLAUDE.md
§3.1 (probe-before-adapter) ahead of the Phase-2 SKILL, which does not exist
yet. Conventions measured by `scripts/probe_wan.py`; findings in
`docs/probe_wan_report.md`. Do not wire this into a stage until the Phase-2
SKILL freezes its contract (§3.5).

Origin: https://github.com/ditflow/ditflow (CVPR 2025), CogVideoX-only. The
port is self-contained -- it shares no code with the CogVideoX implementation.
"""

from .attention import WanInjectionProcessor, WanModuleWithGuidance, apply_rotary_emb
from .motion_flow import compute_motion_flow
from .transformer import ControlledWanTransformer

__all__ = [
    "ControlledWanTransformer",
    "WanInjectionProcessor",
    "WanModuleWithGuidance",
    "apply_rotary_emb",
    "compute_motion_flow",
]
