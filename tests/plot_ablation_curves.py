import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
runs=[('MambaIRv2 (baseline)','MambaIRv2_SR_x2_50k','#2a78d6'),
      ('GatedMambaIR · GLA','GatedMambaIR_SR_x2_gla_50k','#eb6834'),
      ('GatedMambaIR · delta','GatedMambaIR_SR_x2_delta_50k','#1baf7a')]
data={}
for lab,d,c in runs:
    ea=EA('tb_logger/'+d,size_guidance={'scalars':0}); ea.Reload(); tags=ea.Tags()['scalars']
    pt=[t for t in tags if 'psnr' in t.lower()][0]; lt=[t for t in tags if 'l_pix' in t][0]
    data[lab]=([(e.step,e.value) for e in ea.Scalars(pt)],[(e.step,e.value) for e in ea.Scalars(lt)],c)
    print(lab,pt,lt,len(data[lab][0]),len(data[lab][1]))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8a85','axes.labelcolor':'#3d3d3a','xtick.color':'#5f5e5a','ytick.color':'#5f5e5a'})
fig,ax=plt.subplots(1,2,figsize=(13,5),facecolor='white')
for lab,(p,l,c) in data.items():
    s,v=zip(*p); s=np.array(s)/1000
    ax[0].plot(s,v,color=c,lw=2,marker='o',ms=6,mec='white',mew=1.5,label=lab)
    ax[0].annotate(f'{v[-1]:.2f}',(s[-1],v[-1]),xytext=(6,0),textcoords='offset points',va='center',color='#3d3d3a',fontsize=10)
    s,v=zip(*l); s=np.array(s); v=np.array(v); k=max(1,len(v)//50)
    sm=np.convolve(v,np.ones(k)/k,mode='valid'); ax[1].plot(s[k-1:]/1000,sm,color=c,lw=2,label=lab)
ax[0].set(title='Set14 validation PSNR (higher is better)',xlabel='training iterations (k)',ylabel='PSNR (dB)',ylim=(31.5,33.7))
ax[1].set(title='Training L1 loss, smoothed (lower is better)',xlabel='training iterations (k)',ylabel='l_pix',yscale='log')
for a in ax: a.grid(axis='y',color='#e6e5e0',lw=0.8); a.set_xlim(0,54); a.title.set_color('#1a1a19')
ax[0].legend(frameon=False,loc='lower right')
fig.text(0.5,-0.02,'×2 SR, 50k-iteration ablation, batch 2 on V100. Delta run still training.',ha='center',color='#5f5e5a',fontsize=10)
fig.tight_layout(); fig.savefig('docs/ablation50k_curves.png',dpi=130,bbox_inches='tight')
