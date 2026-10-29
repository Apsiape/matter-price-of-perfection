"""Check source consistency, abstract length, line endings, and metadata."""
import json
from pathlib import Path
import re
import subprocess
import sys

here=Path(__file__).resolve().parent
main=(here/'main.tex').read_text(encoding='utf-8')
abstract=' '.join(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1).split())
assert len(abstract)<=1920
assert (here/'abstract.txt').read_text(encoding='utf-8').strip()==abstract
sections=re.findall(r'\\input\{([^}]+)\}',main)
for name in sections:assert (here/(name+'.tex')).is_file(),name
alltex='\n'.join(p.read_text(encoding='utf-8') for p in here.glob('*.tex'))
labels=re.findall(r'\\label\{([^}]+)\}',alltex)
assert len(labels)==len(set(labels))
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',alltex)
assert set(refs)<=set(labels)
bib=(here/'references.bib').read_text(encoding='utf-8')
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
cites=[]
for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',alltex):cites+=group.split(',')
assert set(cites)==keys
for p in here.rglob('*'):
    if p.is_file() and p.suffix not in ('.pdf','.png'):
        assert b'\r' not in p.read_bytes(),str(p.relative_to(here))
subprocess.run([sys.executable,str(here/'anc/verify_hashes.py')],check=True)
print(json.dumps(dict(status='PASS',abstract_characters=len(abstract),sections=len(sections),
                     labels=len(labels),bibliography_entries=len(keys),line_endings='LF')))
