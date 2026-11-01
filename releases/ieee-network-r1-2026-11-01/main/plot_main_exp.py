"""Reproduce the GROUND_PARALLEL_V1 revised IEEE Network main figure from packaged full-precision CSV.

Usage: python plot_main_exp.py [--output-dir PATH]
Dependencies: numpy, matplotlib, Pillow, pypdf. No experiment/runtime imports.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import os

# Tiny plotting linear algebra needs one thread; keep this local to this process.
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch, Circle
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker, TextArea
from matplotlib.text import Text
from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
METHODS = ['GROUND', 'OFFLOAD', 'TUT_ADAPTED', 'PROP']
LABELS = ['GROUND', 'OFFLOAD', 'TUT-Adapt', 'PROP']
COLORS = ['#78B6C8', '#F2D98E', '#466F87', '#E48578']
LIGHT = [tuple(.65*np.array(to_rgb(c))+.35) for c in COLORS]
PRODUCTS = ['metadata', 'roi', 'roi_context']
PCOLORS = ['#DCE9EE', '#78B6C8', '#466F87']
PARTIAL_HATCH = '///'  # 50% denser than V3; same 0.40 pt stroke.
TEXT = '#30343A'
EDGE = '#555B61'
GRID = '#E4E6E8'
WIDTH_MM, HEIGHT_MM = 182, 132
BUBBLE_SCALE = 1.0  # Matplotlib s (points^2) = this common factor * overall net utility.


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data():
    with (HERE/'plot_input_per_seed.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    expected = {(m,s) for m in METHODS for s in range(4101,4106)}
    assert len(records) == 20 and {(r['method'],int(r['seed'])) for r in records} == expected
    for r in records:
        for k in r:
            if k != 'method': r[k] = float(r[k])
        assert math.isclose(r['overall_net'], r['hv_net']+r['ordinary_net'], abs_tol=1e-8)
        assert r['total_closed'] == r['hv_closed']+r['ordinary_closed']
        assert sum(r[p+'_satisfied'] for p in PRODUCTS) == r['total_closed']
        assert sum(r[p+'_partial'] for p in PRODUCTS) == r['partial_only']
        assert r['total_closed']+r['partial_only'] == r['effective_delivered']
    metrics = [k for k in records[0] if k not in ['method','seed']]
    aggregate = []
    for i,m in enumerate(METHODS):
        rr = [r for r in records if r['method']==m]
        for key in metrics:
            vals = [r[key] for r in rr]
            aggregate.append(dict(method=m, display_method=LABELS[i], metric=key, n=5, mean=statistics.mean(vals), sd=statistics.stdev(vals), ddof=1))
        for p in PRODUCTS:
            vals = [r[p+'_satisfied']+r[p+'_partial'] for r in rr]
            aggregate.append(dict(method=m,display_method=LABELS[i],metric=p+'_total',n=5,mean=statistics.mean(vals),sd=statistics.stdev(vals),ddof=1))
    stat = {(r['method'],r['metric']):(r['mean'],r['sd']) for r in aggregate}
    return stat, aggregate

def configure():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'] if f in names), 'DejaVu Serif')
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':[family], 'font.size':8.5,
        'mathtext.fontset':'stix', 'text.color':TEXT,
        'axes.labelcolor':TEXT, 'axes.edgecolor':EDGE, 'axes.linewidth':.6,
        'axes.labelsize':9, 'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
        'xtick.color':TEXT, 'ytick.color':TEXT, 'xtick.major.width':.6,
        'ytick.major.width':.6, 'xtick.major.size':2.5, 'ytick.major.size':2.5,
        'legend.fontsize':8, 'legend.frameon':True, 'legend.fancybox':False,
        'legend.facecolor':'white', 'legend.edgecolor':EDGE, 'legend.framealpha':1,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'path',
        'hatch.linewidth':.40, 'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white', 'savefig.transparent':False,
        'axes.unicode_minus':False, 'svg.hashsalt':'network_frozen_main_20_v1',
    })
    return family

def decorate(ax, title, orientation='vertical'):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.6)
    ax.set_axisbelow(True)
    ax.grid(axis='both', color=GRID, linewidth=.45)
    ax.tick_params(which='both', direction='in', top=True, right=True, pad=3)
    caption_y=-.18 if title.startswith(('(a)','(b)')) else -.28
    ax.text(.5,caption_y,title,transform=ax.transAxes,ha='center',va='top',fontsize=9.5,fontweight='bold',clip_on=False)

def framed_legend(ax, handles, **kwargs):
    legend=ax.legend(handles=handles,frameon=True,fancybox=False,shadow=False,
                     facecolor='white',edgecolor=EDGE,framealpha=1,
                     borderpad=.4,handletextpad=.45,handlelength=1.25,
                     labelspacing=.35,**kwargs)
    legend.get_frame().set_linewidth(.6)
    ax._publication_legend=legend
    return legend

def stats(stat, key):
    return (np.array([stat[m,key][0] for m in METHODS]), np.array([stat[m,key][1] for m in METHODS]))

def error(ax, x, y, xerr=None, yerr=None):
    return ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',ecolor=TEXT,elinewidth=.8,capsize=2.0,capthick=.8,zorder=5)

def panel_a(ax, st):
    decorate(ax, '(a) Net utility')
    x = np.arange(4)
    for offset,key,colors in [(-.20,'overall_net',LIGHT),(.20,'hv_net',COLORS)]:
        v,sd = stats(st,key)
        ax.bar(x+offset,v,width=.34,color=colors,edgecolor=EDGE,linewidth=.45,zorder=3)
        error(ax,x+offset,v,yerr=sd)
        for xx,yy,ss in zip(x+offset,v,sd):
            ax.annotate(f'{yy:.1f}',(xx,yy+ss),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,LABELS)
    ax.set_xlim(-.62,3.62); ax.set_ylim(0,455)
    ax.set_yticks(np.arange(0,401,100)); ax.set_ylabel('Net utility',labelpad=5)
    framed_legend(ax,[Patch(facecolor='#CCD0D3',edgecolor=EDGE,linewidth=.45,label='Overall'),Patch(facecolor='#808991',edgecolor=EDGE,linewidth=.45,label='HV')],loc='upper left',ncol=2,borderaxespad=.65,columnspacing=1.0)

def panel_b(ax, st):
    decorate(ax, '(b) Task closure')
    x=np.arange(4); hv,hs=stats(st,'hv_closed'); od,_=stats(st,'ordinary_closed'); total,ts=stats(st,'total_closed')
    ax.bar(x,hv,width=.44,color=COLORS,edgecolor=EDGE,linewidth=.45,zorder=3)
    ax.bar(x,od,bottom=hv,width=.44,color=LIGHT,edgecolor=EDGE,linewidth=.45,zorder=3)
    error(ax,x,hv,yerr=hs); error(ax,x,total,yerr=ts)
    for i in range(4):
        ax.text(i,hv[i]/2,f'{hv[i]:.1f}',ha='center',va='center',fontsize=8.5,color='white' if i==2 else TEXT)
        ax.text(i,hv[i]+od[i]/2,f'{od[i]:.1f}',ha='center',va='center',fontsize=8.5)
        ax.annotate(f'{total[i]:.1f}',(i,total[i]+ts[i]),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,LABELS); ax.set_xlim(-.62,3.62); ax.set_ylim(0,max(815,math.ceil((max(total+ts)+90)/100)*100))
    ax.set_yticks(np.arange(0,ax.get_ylim()[1],400 if ax.get_ylim()[1]>1200 else 200)); ax.set_ylabel('Closed tasks',labelpad=5)
    framed_legend(ax,[Patch(facecolor='#808991',edgecolor=EDGE,linewidth=.45,label='HV'),Patch(facecolor='#CCD0D3',edgecolor=EDGE,linewidth=.45,label='Ordinary')],loc='upper right' if ax.get_ylim()[1]>1000 else 'upper left',ncol=1,borderaxespad=.65)

def panel_c(ax, st):
    decorate(ax, '(c) Ground product availability', 'horizontal')
    left=np.zeros(4)
    ax._product_labels=[]
    for p,color in zip(PRODUCTS,PCOLORS):
        satisfied,_=stats(st,p+'_satisfied'); partial,_=stats(st,p+'_partial'); total=satisfied+partial
        for i in range(4):
            if satisfied[i]>0:
                ax.barh(i,satisfied[i],left=left[i],height=.40,color=color,edgecolor=EDGE,linewidth=.4,zorder=3)
            if partial[i]>0:
                ax.barh(i,partial[i],left=left[i]+satisfied[i],height=.40,color=color,edgecolor=EDGE,linewidth=.4,hatch=PARTIAL_HATCH,zorder=3)
            if total[i]>0:
                # Label the entire product segment above the bar, preserving
                # both its solid and partial-only hatch without a masking box.
                label=ax.annotate(f'{total[i]:.1f}',(left[i]+total[i]/2,i-.20),
                                  xytext=(0,1.8),textcoords='offset points',
                                  ha='center',va='bottom',fontsize=8.2,color=TEXT,zorder=6)
                ax._product_labels.append(label)
        left+=total
    total,sd=stats(st,'effective_delivered')
    assert np.allclose(left,total)
    error(ax,total,np.arange(4),xerr=sd)
    for i,(value,s) in enumerate(zip(total,sd)):
        ax.annotate(f'{value:.1f}\n±{s:.1f}',(value+s,i),xytext=(4,0),textcoords='offset points',ha='left',va='center',fontsize=8,linespacing=1.05)
    ax.set_yticks(np.arange(4),LABELS); ax.set_ylim(3.55,-.70)
    # Reserve a right-hand band for the user-requested vertical legend.
    ax.set_xlim(0,max(2000,math.ceil((max(total+sd)+700)/200)*200)); ax.set_xticks(np.arange(0,ax.get_xlim()[1]+1,500)); ax.set_xlabel('Tasks with an effective ground product',labelpad=4)
    handles=[Patch(facecolor=c,edgecolor=EDGE,linewidth=.4,label=label) for c,label in zip(PCOLORS,['Metadata','ROI','ROI+Ctx'])]
    handles.append(Patch(facecolor='#FFFFFF',edgecolor=EDGE,linewidth=.4,hatch=PARTIAL_HATCH,label='Partial only'))
    framed_legend(ax,handles,loc='center right',bbox_to_anchor=(.99,.40),ncol=1,borderaxespad=.35)

def panel_d(ax, st):
    decorate(ax, '(d) Resource–net utility tradeoff')
    en,es=stats(st,'controllable_energy_kj'); mb,ms=stats(st,'downlink_mb'); util,_=stats(st,'overall_net')
    assert np.all(util>0)
    ax.set_xlim(min(30,math.floor(min(en-es-12)/10)*10),max(75,math.ceil(max(en+es+10)/10)*10)); ax.set_ylim(0,max(3000,math.ceil(max(mb+ms+300)/500)*500))
    ax.set_xticks(np.arange(ax.get_xlim()[0],ax.get_xlim()[1]+1,20)); ax.set_yticks(np.arange(0,ax.get_ylim()[1]+1,1000 if ax.get_ylim()[1]>3500 else 500))
    ax.set_xlabel('Controllable energy (kJ)',labelpad=4); ax.set_ylabel('Downlink volume (MB)',labelpad=5)
    ax.scatter(en,mb,s=BUBBLE_SCALE*util,c=COLORS,edgecolors=EDGE,linewidths=.55,zorder=3)
    error(ax,en,mb,xerr=es,yerr=ms)
    ax._bubble_labels=[]
    for x,y,sd,utility,label in zip(en,mb,ms,util,LABELS):
        radius_pt=math.sqrt(BUBBLE_SCALE*utility)/2+.55/2
        error_pt=sd/ax.get_ylim()[1]*ax.bbox.height/ax.figure.dpi*72
        dx={'TUT-Adapt':-6.0,'PROP':6.0}.get(label,0)
        annotation=ax.annotate(label,(x,y),xytext=(dx,-max(radius_pt,error_pt)-2),
                               textcoords='offset points',ha='center',va='top',fontsize=8.5,zorder=7)
        ax._bubble_labels.append(annotation)
    # Compact variable-height rows; circle diameters exactly match scatter's s.
    # DrawingArea dimensions are points, so legend and plotted markers share scale.
    rows=[]
    for value in [25,100,300]:
        diameter=math.sqrt(BUBBLE_SCALE*value)
        height=max(diameter,8.0)
        marker=DrawingArea(22,height,0,0)
        marker.add_artist(Circle((11,height/2),diameter/2,facecolor='white',edgecolor=EDGE,linewidth=.55))
        number=TextArea(str(value),textprops=dict(fontsize=8,color=TEXT))
        rows.append(HPacker(children=[marker,number],align='center',pad=0,sep=5))
    title=TextArea('Area: overall\nnet utility',textprops=dict(fontsize=8,color=TEXT,ha='center',multialignment='center',linespacing=1.0))
    body=VPacker(children=[title,VPacker(children=rows,align='left',pad=0,sep=1.5)],align='center',pad=0,sep=2)
    box=AnchoredOffsetbox(loc='upper left',child=body,pad=.35,borderpad=.65,frameon=True,prop=dict(size=8))
    box.patch.set_facecolor('white'); box.patch.set_edgecolor(EDGE); box.patch.set_linewidth(.6)
    ax.add_artist(box)
    ax._publication_legend=box

def check_legend(ax,key,st):
    renderer=ax.figure.canvas.get_renderer()
    box=ax._publication_legend.get_window_extent(renderer)
    overlaps=[t.get_text() for t in ax.texts if box.overlaps(t.get_window_extent(renderer))]
    assert not overlaps, ('LEGEND_OVERLAPS_LABEL',key,overlaps)
    patch_overlaps=int(sum(box.overlaps(p.get_window_extent(renderer)) for p in ax.patches))
    assert patch_overlaps==0, ('LEGEND_OVERLAPS_BAR',key)
    bubble_gaps=[]
    if key=='c':
        label_boxes=[label.get_window_extent(renderer) for label in ax._product_labels]
        for i,(label,label_box) in enumerate(zip(ax._product_labels,label_boxes)):
            assert ax.bbox.contains(label_box.x0,label_box.y0) and ax.bbox.contains(label_box.x1,label_box.y1), ('PRODUCT_LABEL_OUTSIDE_AXES',label.get_text())
            assert not any(label_box.overlaps(p.get_window_extent(renderer)) for p in ax.patches), ('PRODUCT_LABEL_OVERLAPS_BAR',label.get_text())
            assert not any(label_box.overlaps(other) for other in label_boxes[i+1:]), ('PRODUCT_LABEL_OVERLAP',label.get_text())
    if key=='d':
        for method in METHODS:
            x,y=ax.transData.transform((st[method,'controllable_energy_kj'][0],st[method,'downlink_mb'][0]))
            dx=max(box.x0-x,0,x-box.x1); dy=max(box.y0-y,0,y-box.y1)
            radius=(math.sqrt(BUBBLE_SCALE*st[method,'overall_net'][0])/2+.55/2)*ax.figure.dpi/72
            gap=(math.hypot(dx,dy)-radius)/ax.figure.dpi*72
            assert gap>1, ('LEGEND_TOO_CLOSE_TO_BUBBLE',method,gap)
            bubble_gaps.append(dict(method=method,clearance_pt=gap))
        label_boxes=[label.get_window_extent(renderer) for label in ax._bubble_labels]
        for i,(method,label_box) in enumerate(zip(METHODS,label_boxes)):
            center=ax.transData.transform((st[method,'controllable_energy_kj'][0],st[method,'downlink_mb'][0]))
            radius=(math.sqrt(BUBBLE_SCALE*st[method,'overall_net'][0])/2+.55/2)*ax.figure.dpi/72
            assert label_box.y1<center[1]-radius, ('LABEL_NOT_BELOW_BUBBLE',method)
            assert ax.bbox.contains(label_box.x0,label_box.y0) and ax.bbox.contains(label_box.x1,label_box.y1), ('LABEL_OUTSIDE_AXES',method)
            for other in label_boxes[i+1:]:
                assert not label_box.overlaps(other), ('BUBBLE_LABEL_OVERLAP',method)
    return dict(panel=key,label_overlaps=overlaps,bar_overlaps=patch_overlaps,legend_to_bubble_gaps=bubble_gaps)

def export(fig, stem):
    # No tight-bbox resizing: preserve the declared physical insertion size.
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':stem.name,'Creator':'Matplotlib; validated GROUND_PARALLEL_V1 results','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')

def pdf_audit(path):
    reader=PdfReader(path)
    assert len(reader.pages)==1
    fonts=[]
    page=reader.pages[0]
    for ref in page['/Resources']['/Font'].get_object().values():
        f=ref.get_object(); descendants=f.get('/DescendantFonts',[f])
        for child in descendants:
            child=child.get_object(); descriptor=child.get('/FontDescriptor')
            embedded=False
            if descriptor:
                desc=descriptor.get_object(); embedded=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
            fonts.append(dict(name=str(f.get('/BaseFont')),subtype=str(f.get('/Subtype')),embedded=embedded))
    assert fonts and all(f['embedded'] for f in fonts)
    images=[]
    for name,ref in page['/Resources'].get('/XObject',{}).items():
        if ref.get_object().get('/Subtype')=='/Image': images.append(str(name))
    assert not images, 'PDF should remain vector'
    return dict(file=path.name, width_mm=float(page.mediabox.width)/72*25.4,height_mm=float(page.mediabox.height)/72*25.4,fonts=fonts,raster_images=images)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    args=parser.parse_args(); out=args.output_dir.resolve(); out.mkdir(exist_ok=True,parents=True)
    st, aggregate=load_data(); family=configure()
    write_csv(out/'plot_mean_sd.csv',aggregate)
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=150)
    axes=[fig.add_axes(b) for b in [[.107,.625,.382,.34],[.594,.625,.382,.34],[.107,.15,.382,.34],[.594,.15,.382,.34]]]
    draw=[panel_a,panel_b,panel_c,panel_d]
    for fun,ax in zip(draw,axes): fun(ax,st)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5:
                outside.append(t.get_text())
    assert not outside, ('TEXT_OUTSIDE_CANVAS',outside)
    legend_checks=[check_legend(ax,key,st) for key,ax in zip('abcd',axes)]
    # Verify real bubble footprints do not overlap in display coordinates.
    positions=axes[3].transData.transform([[st[m,'controllable_energy_kj'][0],st[m,'downlink_mb'][0]] for m in METHODS])
    radii=np.sqrt([BUBBLE_SCALE*st[m,'overall_net'][0] for m in METHODS])/2*fig.dpi/72
    clearances=[]
    for i in range(4):
        for j in range(i+1,4):
            clearance=float(np.linalg.norm(positions[i]-positions[j])-radii[i]-radii[j]-.55*fig.dpi/72)
            assert clearance>0
            clearances.append(dict(method1=METHODS[i],method2=METHODS[j],clearance_mm=clearance/fig.dpi*25.4))
    export(fig,out/'main_exp')
    fig.savefig(out/'main_exp_preview.png',dpi=180)
    fig.savefig(out/'main_exp_size_check_150dpi.png',dpi=150)
    plt.close(fig)
    subdir=out/'panels'; subdir.mkdir(exist_ok=True)
    for key,fun in zip(['a_net_utility','b_task_closure','c_ground_products','d_resource_use'],draw):
        subfig=plt.figure(figsize=(91/25.4,66/25.4),dpi=150)
        # Match the physical axes size in the full figure exactly (69.524 x 44.88 mm).
        ax=subfig.add_axes([.20,.30,.764,.68])
        fun(ax,st)
        subfig.canvas.draw()
        renderer=subfig.canvas.get_renderer()
        for text in subfig.findobj(Text):
            if text.get_visible() and text.get_text():
                bounds=text.get_window_extent(renderer)
                assert bounds.x0>=-.5 and bounds.y0>=-.5 and bounds.x1<=subfig.bbox.width+.5 and bounds.y1<=subfig.bbox.height+.5, ('STANDALONE_TEXT_CROPPED',key,text.get_text())
        standalone_check=check_legend(ax,key[0],st)
        standalone_check['scope']='standalone'
        legend_checks.append(standalone_check)
        export(subfig,subdir/key)
        plt.close(subfig)
    audits=[pdf_audit(p) for p in [out/'main_exp.pdf',*sorted(subdir.glob('*.pdf'))]]
    im=Image.open(out/'main_exp.png')
    assert abs(im.info['dpi'][0]-600)<.1 and im.getpixel((0,0))[:3]==(255,255,255)
    write_json(out/'EXPORT_CHECKS.json',dict(font_family=family,font_file=font_manager.findfont(family),font_fallback=family!='Times New Roman',main_png_pixels=list(im.size),main_png_dpi=im.info['dpi'],main_figure_mm=[WIDTH_MM,HEIGHT_MM],minimum_text_pt=8,axis_linewidth_pt=.6,bar_edge_linewidth_pt=.45,error_linewidth_pt=.8,hatch_linewidth_pt=.4,hatch=PARTIAL_HATCH,method_colors=COLORS,light_colors=LIGHT,light_formula='0.65 * base RGB + 0.35 * white RGB',product_colors=PCOLORS,bubble_scale_points2_per_net_utility=BUBBLE_SCALE,bubble_area_offset=0,bubble_clearances=clearances,text_outside_canvas=outside,legend_checks=legend_checks,style_revision=dict(b_bar_width=.44,c_bar_height=.40,all_spines_visible=True,ticks='in; all four sides',grid='both; light gray',panel_titles='below axes',legend_d='framed vertical, exact marker size'),pdfs=audits,versions=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__),input_csv_sha256=sha(HERE/'plot_input_per_seed.csv'),visual_review='PENDING_ACTUAL_IMAGE_INSPECTION'))
    print(json.dumps({'status':'EXPORTED','font':family,'figure_mm':[WIDTH_MM,HEIGHT_MM],'png_px':list(im.size),'pdf_fonts_embedded':True,'bubble_overlap':False,'out':str(out)},ensure_ascii=False))

if __name__=='__main__':
    main()
