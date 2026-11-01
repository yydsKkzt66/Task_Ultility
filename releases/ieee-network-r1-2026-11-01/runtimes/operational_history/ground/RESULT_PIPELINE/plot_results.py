"""Updated four-panel figure; explicit partial preview until all20 runs exist."""
from pathlib import Path
import json
import math
import os
import shutil
import statistics
import sys

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
# Standalone plotting needs the same installed Conda DLL search path, without
# importing or executing the simulator. Detached updater already inherits it.
_dll_handles=[]
if os.name=='nt':
    dll=Path(sys.prefix)/'Library/bin'
    if dll.exists():
        os.environ['PATH']=str(dll)+os.pathsep+os.environ.get('PATH','')
        _dll_handles.append(os.add_dll_directory(str(dll)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pypdf import PdfReader
from collect_results import collect,OUT,write,sha,METHODS

COLORS=['#78B6C8','#F2D98E','#466F87','#E48578']
LABELS=['GROUND','OFFLOAD','TUT-Adapt','PROP']
PRODUCTS=['metadata','roi','roi_context']
PCOLORS=['#DCE9EE','#78B6C8','#466F87']


def draw(data,state):
    folder=OUT/'GENERATIONS'/(state['signature'][:12]+'_'+sha(__file__)[:8])
    done=folder/'PLOT_VERIFIED.json'
    if done.exists():
        verification=read_done(done)
        for name in verification['files']:
            shutil.copyfile(folder/name,OUT/('main_exp_latest'+Path(name).suffix))
        write(OUT/'LATEST_FIGURE.json',verification)
        return verification
    folder.mkdir(parents=True,exist_ok=True)
    complete=state['ready_for_formal_figure']
    counts=[sum(r['method']==m for r in data) for m in METHODS]
    labels=[f'{label}\n(n={n}/5)' if not complete else label for label,n in zip(LABELS,counts)]
    rows={m:[r for r in data if r['method']==m] for m in METHODS}
    mean=lambda m,k:statistics.mean(r[k] for r in rows[m])
    sd=lambda m,k:statistics.stdev(r[k] for r in rows[m]) if len(rows[m])==5 else 0.
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','figure.facecolor':'white',
        'hatch.linewidth':.5,'axes.spines.top':True,'axes.spines.right':True})
    fig,axs=plt.subplots(2,2,figsize=(9.2,7.0))
    fig.subplots_adjust(left=.085,right=.985,bottom=.105,top=.91,wspace=.31,hspace=.43)
    fig.suptitle('Updated main experiment: five paired seeds' if complete else
        f'Audited results in progress - {state["accepted"]}/20 runs complete',fontsize=12,y=.98)
    for ax in axs.flat:
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E6E8',linewidth=.5)
        ax.tick_params(direction='in',top=True,right=True)
    a,b,c,d=axs.flat
    a.set_title('(a) Realized net utility',loc='left',pad=10)
    for i,m in enumerate(METHODS):
        if not rows[m]:continue
        for offset,key,alpha in [(-.2,'overall_net',.48),(.2,'hv_net',1.)]:
            y=mean(m,key);a.bar(i+offset,y,.34,color=COLORS[i],alpha=alpha,edgecolor='#555',linewidth=.5)
            if len(rows[m])==5:a.errorbar(i+offset,y,yerr=sd(m,key),color='#333',capsize=2,linewidth=.7)
            a.annotate(f'{y:.1f}',(i+offset,y+sd(m,key)),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    a.set_xticks(range(4),labels);a.set_ylabel('Net utility')
    a.legend(handles=[Patch(facecolor='#aaa',alpha=.48,label='Overall'),Patch(facecolor='#777',label='HV')],loc='upper left',ncol=2)
    a.margins(y=.21);a.axhline(0,color='#555',lw=.5)
    b.set_title('(b) Task closure',loc='left',pad=10)
    for i,m in enumerate(METHODS):
        if not rows[m]:continue
        hv,ordinary=mean(m,'hv_closed'),mean(m,'ordinary_closed');total=hv+ordinary
        b.bar(i,hv,.48,color=COLORS[i],edgecolor='#555',linewidth=.5)
        b.bar(i,ordinary,.48,bottom=hv,color=COLORS[i],alpha=.42,edgecolor='#555',linewidth=.5)
        b.text(i,hv/2,f'{hv:.0f}',ha='center',va='center',fontsize=8,color='white' if i==2 else '#333')
        b.text(i,hv+ordinary/2,f'{ordinary:.0f}',ha='center',va='center',fontsize=8)
        if len(rows[m])==5:b.errorbar(i,total,yerr=sd(m,'total_closed'),color='#333',capsize=2,linewidth=.7)
        b.annotate(f'{total:.1f}',(i,total+sd(m,'total_closed')),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    b.set_xticks(range(4),labels);b.set_ylabel('Closed tasks');b.set_ylim(0,1600)
    b.legend(handles=[Patch(facecolor='#777',label='HV'),Patch(facecolor='#bbb',label='Ordinary')],loc='upper right',ncol=2)
    c.set_title('(c) Highest valid product on ground',loc='left',pad=10)
    for i,m in enumerate(METHODS):
        if not rows[m]:continue
        left=0.
        for p,color in zip(PRODUCTS,PCOLORS):
            for kind,hatch in [('satisfied',None),('partial','///')]:
                n=mean(m,p+'_'+kind)
                if n:
                    c.barh(i,n,.45,left=left,color=color,hatch=hatch,edgecolor='#555',linewidth=.45)
                    if n>80:c.text(left+n/2,i,f'{n:.0f}',ha='center',va='center',fontsize=7,color='white' if p=='roi_context' else '#333')
                left+=n
        c.text(left+15,i,f'{left:.1f}',va='center',fontsize=8)
    c.set_yticks(range(4),labels);c.invert_yaxis();c.set_xlim(0,1600);c.set_xlabel('Tasks (each counted once)')
    c.grid(False);c.grid(axis='x',color='#E4E6E8',linewidth=.5)
    c.legend(handles=[Patch(facecolor=color,edgecolor='#555',label=label) for color,label in zip(PCOLORS,['Metadata','ROI','ROI+Ctx'])]+
             [Patch(facecolor='white',edgecolor='#555',hatch='///',label='Partial')],ncol=2,loc='lower right',fontsize=7)
    c.set_ylim(4.3,-.65)
    d.set_title('(d) Controllable resource use',loc='left',pad=10)
    for i,m in enumerate(METHODS):
        if not rows[m]:continue
        x=[r['controllable_energy_kj'] for r in rows[m]];y=[r['downlink_mb'] for r in rows[m]]
        d.scatter(x,y,s=22,color=COLORS[i],edgecolor='#555',linewidth=.45,label=f'{LABELS[i]} (n={len(x)})',zorder=3)
        if len(x)==5:
            d.errorbar(statistics.mean(x),statistics.mean(y),xerr=statistics.stdev(x),yerr=statistics.stdev(y),
                fmt='D',ms=4,color=COLORS[i],ecolor='#555',elinewidth=.65,capsize=2,zorder=4)
    d.set_xlabel('Controllable energy (kJ)');d.set_ylabel('Actual downlink (MB)')
    d.legend(loc='upper left',fontsize=7);d.margins(x=.15,y=.15);d.yaxis.set_major_locator(MaxNLocator(5))
    note=('Mean and sample SD (ddof=1), five seeds per method; dots in (d) show individual seeds.' if complete else
          'Preview only: bars average available audited seeds (n shown); SD only for n=5. Missing runs are excluded.')
    fig.text(.5,.025,note,ha='center',fontsize=8)
    stem='main_exp' if complete else 'main_exp_progress'
    for suffix in ['pdf','png','svg']:
        fig.savefig(folder/(stem+'.'+suffix),dpi=220,facecolor='white')
    plt.close(fig)
    pdf=PdfReader(folder/(stem+'.pdf'))
    assert len(pdf.pages)==1 and all(label in pdf.pages[0].extract_text() for label in LABELS)
    verification=dict(signature=state['signature'],complete_five_seed_figure=complete,n_by_method=dict(zip(METHODS,counts)),
        no_failed_or_partial_runs=True,one_pdf_page=True,legend_methods_present=True,manual_visual_review='PENDING',
        files={stem+'.'+suffix:sha(folder/(stem+'.'+suffix)) for suffix in ['pdf','png','svg']},
        generation=str(folder),marker_note='Node unavailable on host and node_repl; supplied pre-authoring marker could not run. Plot generation and PDF content checks use installed Python tools.')
    write(done,verification)
    for suffix in ['pdf','png','svg']:
        shutil.copyfile(folder/(stem+'.'+suffix),OUT/(stem+'.'+suffix))
        shutil.copyfile(folder/(stem+'.'+suffix),OUT/('main_exp_latest.'+suffix))
    write(OUT/'LATEST_FIGURE.json',verification)
    return verification


def read_done(path):return json.loads(path.read_text(encoding='utf-8'))


if __name__=='__main__':
    data,state=collect();print(json.dumps(draw(data,state),ensure_ascii=False,indent=2))
