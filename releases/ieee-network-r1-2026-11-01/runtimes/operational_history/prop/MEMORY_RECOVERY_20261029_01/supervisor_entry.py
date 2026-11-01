from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '.network_formal6h_work'
OLD_TOOLS = HERE.parent / 'RECOVERY_20261029_01/TOOLS'
_bootstrap = runpy.run_path(str(SOURCE / 'run.py'), run_name='_memory_supervisor_environment')
sys.path.insert(0, str(OLD_TOOLS))
sys.path.insert(0, str(HERE))
from io_observer import install
install()
from memory_supervisor import main
main(SOURCE, HERE.parent / 'MAIN_RUNS', OLD_TOOLS)
