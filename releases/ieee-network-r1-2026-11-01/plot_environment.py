"""Portable conda/Windows DLL bootstrap; does not alter scientific/plot code."""
from pathlib import Path
import os, sys, runpy
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
handles=[]
if os.name=='nt':
    dll=Path(sys.prefix)/'Library/bin'
    if dll.is_dir():
        os.environ['PATH']=str(dll)+os.pathsep+os.environ.get('PATH','')
        handles.append(os.add_dll_directory(str(dll)))
script=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')
