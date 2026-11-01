"""Package a small current snapshot, excluding raw logs and checkpoints."""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import sys
import zipfile
from collect_results import HERE,PACK,REPO,OUT,sha,write,read

sys.path.insert(0,str(REPO/'tmp/network_figure_pdf_render'))
import pypdfium2 as pdfium
from PIL import Image,ImageChops

qa=OUT/'QA';prior=qa/'main_exp_progress_render.png'
doc=pdfium.PdfDocument(str(OUT/'main_exp_latest.pdf'));page=doc[0];bitmap=page.render(scale=150/72)
render=bitmap.to_pil().convert('RGB')
same=prior.exists() and ImageChops.difference(render,Image.open(prior).convert('RGB')).getbbox() is None
render.save(qa/'main_exp_latest_render.png');bitmap.close();page.close();doc.close()
v=read(OUT/'LATEST_FIGURE.json')
v.update(manual_visual_review='PASS: same rendered pixels as manually reviewed PDF' if same else 'PENDING_NEW_PIXELS',
         render_engine='PDFium150dpi',render_png_sha256=sha(qa/'main_exp_latest_render.png'),same_pixels_as_reviewed_pdf=same)
write(OUT/'LATEST_FIGURE.json',v);write(Path(v['generation'])/'PLOT_VERIFIED.json',v);write(qa/'VISUAL_REVIEW.json',v)
write(HERE/'EVIDENCE_SNAPSHOT.json',dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
       heartbeat=read(HERE/'HEARTBEAT.json'),figure_state=read(OUT/'STATE.json'),figure=v,
       original_figures=[dict(path=str(p),sha256=sha(p)) for p in [
           REPO/'handoff_out/NETWORK_MAIN_FIGURE_2X2_20261029_07/main_exp.pdf',
           REPO/'handoff_out/NETWORK_MAIN_FIGURE_2X2_CAUSAL_20261030_01/main_exp.pdf']]))
files=[f for f in HERE.iterdir() if f.is_file() and f.suffix in ['.py','.md','.json']]
files += [f for f in OUT.iterdir() if f.is_file() and f.suffix in ['.md','.csv','.json','.pdf','.png','.svg']]
files += [qa/'VISUAL_REVIEW.json',PACK/'OFFLOAD4102_CLOCK_RECOVERY/RECOVERY_RESULT.json',
          PACK/'A_RUNS/4101_GROUND/attempt_01/AUDIT.json',PACK/'A_RUNS/4103_GROUND/attempt_01/AUDIT.json']
records=[dict(path=f.relative_to(PACK).as_posix(),sha256=sha(f),bytes=f.stat().st_size) for f in files]
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(records)
(HERE/'DELIVERY_FILES_SHA256.csv').write_text(s.getvalue(),encoding='utf-8-sig')
target=PACK/'AUTOMATIC_RECOVERY_AND_FIGURE_EVIDENCE.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for f,r in zip(files,records):z.write(f,r['path'])
    z.write(HERE/'DELIVERY_FILES_SHA256.csv','DELIVERY_FILES_SHA256.csv')
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for r in records:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii')
print(json.dumps(dict(zip=str(target),bytes=target.stat().st_size,sha256=sha(target),PDF_render_parity=same),ensure_ascii=False))
