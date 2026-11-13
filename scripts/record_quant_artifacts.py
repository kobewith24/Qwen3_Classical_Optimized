from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,file_digest,write_json

files=[]
for name in ['base-q8_0.gguf','clean-f16.gguf','clean-q8_0.gguf','clean-q4_k_m.gguf']:
    path=ROOT/'weights/gguf'/name
    if not path.exists():continue
    with path.open('rb') as f:magic,version,tensors,metadata=struct.unpack('<4sIQQ',f.read(24))
    assert magic==b'GGUF' and version in (2,3) and tensors==398,name
    files.append({'name':name,'bytes':path.stat().st_size,'sha256':file_digest(path),
                  'gguf_version':version,'tensors':tensors,'metadata_fields':metadata})
assert len(files)==4,'Some quantization outputs are missing'
write_json(ROOT/'reports/gguf-artifacts.json',{'files':files,'source_checkpoint':'clean_2k_seed24','template_policy':'same explicit ChatML for all native comparison models'})
print('All four GGUF artifacts have valid headers and recorded SHA-256 digests.')
