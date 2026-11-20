#!/usr/bin/env python3
"""Compile all Python and execute numerical/safety-boundary regressions, no robot."""
import ast,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for p in root.rglob('*.py'):
    if 'data' not in p.relative_to(root).parts:compile(p.read_text(),str(p),'exec')
subprocess.run(['bash','-n',str(root/'run_main.sh')],check=True)
subprocess.run(['bash','-n',str(root/'scripts/with_session.sh')],check=True)
raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],cwd=root))
