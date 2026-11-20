# Initial audit

Base repo fc4e04a; original working tree has only existing .DS_Store change. Independent worktree branch codex/ismr-runnable-v1.

home_to_rcm_insert_v2_precision.py: Drive A == Drive B == local next step/code == attachment: True
run_main.sh: Drive A == Drive B == local next step/code == attachment: True
trial_archiver.py: Drive A == Drive B == local next step/code == attachment: True

Retain existing code by default; proven October updates supersede August equivalents; August fills missing tools only. No firmware changes. Eye-eye tools remain optional in original repository.

Confirmed defects: historical axis override, fixed ring-model sampling dependency, canonical TipInEE bootstrap loop, old visual-correction assumptions, absolute laboratory paths.

Plan: retain flat imports; isolate sessions; phase-specific preflight; preserve control math and chained t/p behavior; document paper differences.
