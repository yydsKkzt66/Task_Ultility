"""Resume a process interruption with the exact checkpoint execution version."""
from pathlib import Path
import argparse
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
PACK=HERE.parent
ROOT=HERE.parents[2]/'.network_ground_parallel_work'
sys.path.insert(0,str(ROOT))
import run
sys.path.insert(0,str(PACK/'RUN_CONTROL_5'))
import formal_common as fc
from runtime_io import write,overlay_identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();previous=a.previous.resolve();out=a.out.resolve()
    assert previous.parent==out.parent and out.parent.parent==(PACK/'A_RUNS').resolve()
    ck=previous/fc.read(previous/'LATEST_CHECKPOINT.json')['path'];meta=fc.read(ck.with_suffix('.json'))
    assert fc.sha(ck)==meta['pickle_sha256'];ids=meta['identity']['manifests'];base=fc.verify()
    assert all(ids.get(k)==v for k,v in base.items())
    ident=fc.read(previous/'IDENTITY.json');overlay=ident.get('execution_overlay')
    if overlay:
        versions={'OFFLOAD_EXACT_ARRIVAL_TAIL_V1':PACK/'OFFLOAD4102_CLOCK_RECOVERY/exact_arrival.py',
                  'GROUND_OFFLOAD_EXACT_ARRIVAL_TAIL_V2':HERE/'exact_arrival.py'}
        version=overlay['version'];path=versions[version]
        assert fc.sha(path)==overlay['sha256']==ids[version] and len(ids)==3
        spec=importlib.util.spec_from_file_location('saved_exact_arrival_version',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install(ROOT)
    else:assert ids==base
    def output(path,value):
        if Path(path).name=='IDENTITY.json':
            value=dict(value,operational_overlay=overlay_identity(),resumed_after_process_interruption=str(previous))
            if overlay:value['execution_overlay']=overlay
        elif Path(path).name=='AUDIT.json' and overlay:
            value=dict(value,execution_overlay=overlay,strict_arrival_identity_unchanged=True)
        write(path,value)
    fc.write=output
    from formal_worker import execute
    from src.network_revision.runtime import NetworkExperiment
    from window_metrics import record
    original=NetworkExperiment.run
    def observed(e,*args,**kw):
        old=e.slot_complete_hook
        def hook(exp,slot):
            old(exp,slot)
            if exp.physics.now in {4200.,9900.,15600.,21300.}:record(exp,out)
        e.slot_complete_hook=hook;record(e,out)
        return original(e,*args,**kw)
    NetworkExperiment.run=observed
    execute(meta['identity']['job'],out,resume=ck,identities=ids)


if __name__=='__main__':main()
