"""Reproduce the complete matched-version three-panel sensitivity figure from CSV.

python plot_sensitivity.py [--output-dir DIR] [--validate-only]
Only reads saved results; no simulator or project-runtime imports.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
METHODS=['GROUND','OFFLOAD','TUT-Adapt','PROP']
SEEDS=[4101,4102,4103]
PANELS={
    'capacity':dict(conditions=['A1','A2','NORMAL','A3'],x=[.15,.30,.50,.90],labels=['0.15','0.30','0.50','0.90'],title='(a) Downlink capacity'),
    'value':dict(conditions=['A4','NORMAL','A5'],x=[0,1,2],labels=['1:2:5:10','1:3:10:25','1:5:20:50'],title='(b) Task-value profile'),
    'service':dict(conditions=['NORMAL','B1','B2'],x=[0,1,2],labels=['Normal','W2 lost','W2+W3 lost'],title='(c) Ground-service interruption')}
METRICS={'Overall':'overall_net','HV':'hv_net'}
MARKERS={'GROUND':'s','OFFLOAD':'D','TUT-Adapt':'^','PROP':'o'}
Y_LIMITS=(-80,450)
FIGURE_MM=(182,82)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def check_close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-8),(a,b)

from validate_data import validate

def configure(style):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    family=next((n for n in [style['font_family'],'Liberation Serif','Nimbus Roman','DejaVu Serif'] if n in names),'DejaVu Serif')
    plt.rcParams.update({
        'font.family':'serif','font.serif':[family],'font.size':9,
        'mathtext.fontset':'stix','text.color':style['TEXT'],
        'axes.labelcolor':style['TEXT'],'axes.edgecolor':style['EDGE'],
        'axes.linewidth':.6,'axes.labelsize':9,
        'xtick.labelsize':9.5,'ytick.labelsize':9.5,
        'xtick.color':style['TEXT'],'ytick.color':style['TEXT'],
        'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,
        'legend.fontsize':9,'legend.frameon':True,'legend.fancybox':False,
        'legend.facecolor':'white','legend.edgecolor':style['EDGE'],'legend.framealpha':1,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'savefig.transparent':False,'axes.unicode_minus':True,
        'svg.hashsalt':'network_sensitivity_complete_three_seed_20261101'})
    return plt,family

def render_figure(data,style,plt):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    colors=dict(zip(style['LABELS'],style['COLORS']))
    assert list(colors)==METHODS
    fig=plt.figure(figsize=(FIGURE_MM[0]/25.4,FIGURE_MM[1]/25.4),dpi=150)
    axes=[];series_records=[]
    for panel_i,(key,panel) in enumerate(PANELS.items()):
        ax=fig.add_axes([.075+panel_i*.31,.24,.287,.60],sharey=axes[0] if axes else None);axes.append(ax)
        for spine in ax.spines.values():spine.set_visible(True)
        ax.set_axisbelow(True);ax.grid(axis='both',color=style['GRID'],linewidth=.45)
        ax.tick_params(which='both',direction='in',top=True,right=True,pad=3,labelleft=panel_i==0)
        ax.set_ylim(*Y_LIMITS);ax.set_yticks([-50,0,100,200,300,400])
        ax.axhline(0,color='#D5D8DC',linewidth=.45,zorder=1.5)
        ax.set_xlim((.105,.945) if key=='capacity' else (-.20,2.30) if key=='service' else (-.30,2.30))
        ax.set_xticks(panel['x'],panel['labels'])
        ax.text(.5,-.155,panel['title'],transform=ax.transAxes,ha='center',va='top',fontsize=9.5,fontweight='bold')
        if panel_i==0:ax.set_ylabel('Net utility',labelpad=5)
        series=[]
        for metric in ['HV','Overall']:
            for method in METHODS:
                rr=[next(r for r in data if r['panel']==key and r['condition']==c and r['method']==method and r['metric']==metric) for c in panel['conditions']]
                x=np.array(panel['x'],float)
                y=np.array([float(r['mean']) if r['draw_in_main'] else np.nan for r in rr])
                sd=np.array([float(r['sample_sd']) if r['draw_in_main'] else np.nan for r in rr])
                linestyle='-' if metric=='Overall' else (0,(3.2,2.2))
                line,=ax.plot(x,y,color=colors[method],linestyle=linestyle,linewidth=1.3 if method=='PROP' else 1.15,zorder=3)
                assert np.isfinite(y).all() and np.isfinite(sd).all(), 'ALL_MATCHED_POINTS_REQUIRED'
                series.append((x,y,sd,method,metric))
                series_records.append(dict(panel=key,method=method,metric=metric,x=x.tolist(),y=[None if np.isnan(v) else float(v) for v in y],color=colors[method],line_style='solid' if metric=='Overall' else 'dashed',marker=MARKERS[method],marker_fill='solid' if metric=='Overall' else 'open'))
        # Errors remain at exact x positions; no dodging or interpolation.
        for x,y,sd,method,metric in series:
            ax.errorbar(x,y,yerr=sd,fmt='none',ecolor=style['TEXT'],elinewidth=.8,capsize=1.8,capthick=.8,zorder=7)
        # Hollow markers are transparent so real coincidences do not mask another point.
        for x,y,sd,method,metric in series:
            ax.plot(x,y,linestyle='none',marker=MARKERS[method],markersize=3.1 if metric=='Overall' else 4.0,
                    markerfacecolor=colors[method] if metric=='Overall' else 'none',
                    markeredgecolor=style['EDGE'] if metric=='Overall' else colors[method],
                    markeredgewidth=.5 if metric=='Overall' else .8,color=colors[method],zorder=6)
    method_handles=[Line2D([],[],linestyle='none',marker=MARKERS[m],markersize=5,color=colors[m],markerfacecolor=colors[m],markeredgecolor=style['EDGE'],markeredgewidth=.55,label=m) for m in METHODS]
    metric_handles=[Line2D([],[],color=style['TEXT'],linewidth=1.15,linestyle='-',marker='o',markersize=3.1,markerfacecolor=style['TEXT'],label='Overall'),
                    Line2D([],[],color=style['TEXT'],linewidth=1.15,linestyle=(0,(3.2,2.2)),marker='o',markersize=4.0,markerfacecolor='none',label='HV')]
    shared_legend=fig.legend(handles=method_handles+metric_handles,loc='upper center',
                            bbox_to_anchor=(.5285,.945),ncol=len(method_handles)+len(metric_handles),
                            borderaxespad=0,borderpad=.45,columnspacing=1.05,handlelength=2.1,handletextpad=.45)
    shared_legend.get_frame().set_linewidth(.6)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[shared_legend.get_window_extent(renderer)]
    assert len(fig.legends)==1,'EXPECTED_ONE_SHARED_LEGEND'
    assert all(not b.overlaps(ax.bbox) for b in boxes for ax in axes),'LEGEND_OVERLAPS_AXES'
    outside=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(renderer)
            if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:outside.append(t.get_text())
    assert not outside,('TEXT_OUTSIDE_CANVAS',outside)
    # All error envelopes lie inside the common y range, with visible negative values.
    assert all(Y_LIMITS[0]<float(r['mean'])-float(r['sample_sd'])<float(r['mean'])+float(r['sample_sd'])<Y_LIMITS[1] for r in data)
    titles=[ax.texts[0].get_window_extent(renderer) for ax in axes]
    assert not any(a.overlaps(b) for i,a in enumerate(titles) for b in titles[i+1:]),'PANEL_TITLES_OVERLAP'
    main_count=sum(len(series['y']) for series in series_records)
    assert main_count==80
    audit=dict(kind='main',figure_mm=list(FIGURE_MM),axes_count=3,layout='1x3',main_points=main_count,restored_ground_points=4,legend_count=1,method_legend_entries=len(method_handles),metric_legend_entries=2,legends_outside_axes=True,legend_overlap=False,text_outside_canvas=outside,panel_title_overlap=False,all_error_envelopes_inside_axes=True,y_limits=list(Y_LIMITS),capacity_x=[.15,.30,.50,.90],categorical_x=[0,1,2],grid_axes='both',grid_color=style['GRID'],grid_linewidth_pt=.45,zero_line_color='#D5D8DC',zero_linewidth_pt=.45,tick_fontsize_pt=9.5,legend_fontsize_pt=9,series=series_records)
    return fig,audit

def export(fig,stem):
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':stem.name,'Creator':'Matplotlib; saved three-seed sensitivity data','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')
    fig.savefig(stem.parent/(stem.name+'_preview.png'),dpi=180)
    fig.savefig(stem.parent/(stem.name+'_size_check_150dpi.png'),dpi=150)

def pdf_audit(path):
    from pypdf import PdfReader
    reader=PdfReader(path);assert len(reader.pages)==1;page=reader.pages[0];fonts=[]
    for ref in page['/Resources']['/Font'].get_object().values():
        f=ref.get_object()
        for child in f.get('/DescendantFonts',[f]):
            child=child.get_object();desc=child.get('/FontDescriptor');embedded=False
            if desc:embedded=any(k in desc.get_object() for k in ['/FontFile','/FontFile2','/FontFile3'])
            fonts.append(dict(name=str(f.get('/BaseFont')),embedded=embedded))
    images=[str(k) for k,v in page['/Resources'].get('/XObject',{}).items() if v.get_object().get('/Subtype')=='/Image']
    assert fonts and all(f['embedded'] for f in fonts) and not images
    return dict(file=path.name,font_embedding=fonts,raster_images=images,width_mm=float(page.mediabox.width)/72*25.4,height_mm=float(page.mediabox.height)/72*25.4)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-csv',type=Path,default=HERE/'inputs/PLOT_NET_UTILITY.csv')
    parser.add_argument('--per-seed-csv',type=Path,default=HERE/'inputs/ALL_96_PER_SEED_METRICS.csv')
    parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    data,validation=validate(args.plot_csv,args.per_seed_csv,out)
    if args.validate_only:print(json.dumps(validation));return
    style=read_json(HERE/'STYLE_FROM_MAIN.json');plt,family=configure(style)
    audits=[]
    fig,audit=render_figure(data,style,plt);export(fig,out/'sensitivity');plt.close(fig);audits.append(audit)
    from PIL import Image
    im=Image.open(out/'sensitivity.png');assert abs(im.info['dpi'][0]-600)<.1 and im.getpixel((0,0))[:3]==(255,255,255)
    pdfs=[pdf_audit(out/(stem+'.pdf')) for stem in ['sensitivity']]
    dump(out/'EXPORT_CHECKS.json',dict(font_family=family,font_fallback=family!='Times New Roman',minimum_text_pt=8.5,tick_fontsize_pt=9.5,legend_fontsize_pt=9,panel_title_pt=9.5,axis_label_pt=9,figure_mm=list(FIGURE_MM),png_pixels=list(im.size),png_dpi=list(im.info['dpi']),method_colors=dict(zip(METHODS,style['COLORS'])),method_markers=MARKERS,error_bars='sample SD; ddof=1; no x displacement',source_style_sha256=style['source_sha256'],pdfs=pdfs,figures=audits,visual_review='PENDING_ACTUAL_INSPECTION'))
    print(json.dumps(dict(status='EXPORTED',font=family,figure_mm=list(FIGURE_MM),main_points=80,excluded_points=0,png_pixels=list(im.size),out=str(out))))

if __name__=='__main__':main()
