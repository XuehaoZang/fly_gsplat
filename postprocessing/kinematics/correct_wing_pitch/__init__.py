"""Diagnostic-only sandbox for the LE/TE straightness-vs-curvature question
(wing pitch `eta` 180-degree wrap crossings). Independent of the production
kinematics modules -- reads `wing_angles.py`/`chord.py`/`pipeline.py` but
never imports/patches their internals, and never writes into them. See
`diag/report.md` for findings.
"""
