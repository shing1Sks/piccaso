import torch, strokegen as sg, numpy as np
torch.set_num_threads(4)
S, keep, labels, ids, anchors = sg.load_data("out/v2","v2")
classes=sorted(set(labels)); ck=torch.load("out/runBnr/ckpt.pt",map_location="cpu")
norm=sg.Norm.__new__(sg.Norm); norm.mean,norm.std=ck["norm"]
m=sg.SetDiT(ck["N"],len(classes)); m.load_state_dict(ck["ema"]); m.eval()
y=torch.tensor([classes.index("house")]*4+[classes.index("sun")]*4)
torch.manual_seed(0); z=sg.sample_B(m,y,ck["N"],25,1.0)
k=z[...,0]
print("keep value histogram over generated slots:", np.histogram(k.flatten().numpy(),bins=[-6,-1.5,-1,-0.5,0,0.5,1,1.5,6])[0].tolist())
print("per level mean keep value (gen):",[round(k[:,a:b].mean().item(),2) for a,b in [(0,16),(16,65),(65,165)]])
dk=(keep.float()*2-1)
print("per level mean keep value (data):",[round(dk[:,a:b].mean().item(),2) for a,b in [(0,16),(16,65),(65,165)]])
# training-time check: the model's x0 prediction from a lightly-noised real sample
idx=torch.tensor([i for i,l in enumerate(labels) if l=="house"][:8])
z0=norm.enc(sg.to_vec(S[idx],keep[idx],anchors),keep[idx]); t=torch.full((8,),0.3); ab=sg.ab_fn(t)[:,None,None]
xt=ab.sqrt()*z0+(1-ab).sqrt()*torch.randn_like(z0)
with torch.no_grad(): v=m(xt,t,torch.full((8,),classes.index("house")))
x0=ab.sqrt()*xt-(1-ab).sqrt()*v
print("denoising a real sample at t=0.3: keep acc", ((x0[...,0]>0)==(z0[...,0]>0)).float().mean().item())
