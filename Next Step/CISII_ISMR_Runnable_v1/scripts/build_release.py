#!/usr/bin/env python3
"""Build the portable source-only laboratory validation ZIP, no real session data."""
import hashlib,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];dist=root.parent/'dist';dist.mkdir(exist_ok=True);target=dist/'CISII_ISMR_Runnable_v1.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or '__pycache__' in rel.parts or p.suffix in ('.pyc','.npz') or '.DS_Store' in rel.parts or rel.parts[:2]==('data','sessions'):continue
        info=zipfile.ZipInfo(str(Path(root.name)/rel),date_time=(2026,11,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(0o100755 if p.suffix=='.sh' else 0o100644)<<16;z.writestr(info,p.read_bytes())
checksum=hashlib.sha256(target.read_bytes()).hexdigest();target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n');print(target);print(checksum)
