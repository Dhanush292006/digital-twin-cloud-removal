from pathlib import Path
import csv, time, random
import torch
from torch.utils.data import DataLoader
from models.proposed_model import ProposedModel
from training.data import SceneDataset
from training.losses import reconstruction_loss


def forward_padded(model,x):
    h,w=x.shape[-2:]; ph=(-h)%16; pw=(-w)%16
    if ph or pw: x=torch.nn.functional.pad(x,(0,pw,0,ph),mode="reflect" if min(h,w)>1 else "replicate")
    return model(x)[...,:h,:w]


def run_training(rows,cfg,output=None):
    seed=int(cfg["seed"]); random.seed(seed); torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda" if cfg.get("device","auto")=="auto" and torch.cuda.is_available() else ("cpu" if cfg.get("device","auto")=="auto" else cfg["device"]))
    features=cfg.get("features")
    train=SceneDataset(rows,"train",features,cfg.get("patch_size",128),augment=True); val=SceneDataset(rows,"validation",features)
    dl=DataLoader(train,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg.get("workers",0))
    vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=cfg.get("workers",0))
    sample=train[0][0]; model=ProposedModel(sample.shape[0],cfg["bands"],cfg["base_channels"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=.5)
    out=Path(output or cfg["checkpoint_dir"]); out.mkdir(parents=True,exist_ok=True)
    history=[]; best=float("inf"); stale=0; start=time.time()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); tl=[]
        for x,y,m,_ in dl:
            x,y,m=x.to(device),y.to(device),m.to(device); opt.zero_grad(set_to_none=True)
            pred=model(x); loss,_=reconstruction_loss(pred,y,m,cfg.get("loss")); loss.backward(); opt.step(); tl.append(float(loss.detach()))
        model.eval(); vloss=[]
        with torch.no_grad():
            for x,y,m,_ in vl:
                loss,_=reconstruction_loss(forward_padded(model,x.to(device)),y.to(device),m.to(device),cfg.get("loss")); vloss.append(float(loss))
        v=float(sum(vloss)/len(vloss)); sched.step(v)
        history.append({"epoch":epoch,"train_loss":sum(tl)/len(tl),"validation_loss":v,"lr":opt.param_groups[0]["lr"]})
        torch.save({"model":model.state_dict(),"config":cfg,"seed":seed,"in_channels":sample.shape[0],"parameters":sum(p.numel() for p in model.parameters()),"features":features},out/"last.pt")
        if v < best:
            best=v; stale=0; torch.save({"model":model.state_dict(),"config":cfg,"seed":seed,"in_channels":sample.shape[0],"parameters":sum(p.numel() for p in model.parameters()),"features":features},out/"best.pt")
        else: stale+=1
        if stale>=cfg["patience"]: break
    with (out/"training_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    import platform
    return {"checkpoint":str(out/"best.pt"),"epochs":len(history),"seconds":time.time()-start,"device":str(device),"gpu_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,"python":platform.python_version(),"pytorch":torch.__version__,"parameters":sum(p.numel() for p in model.parameters()),"features":features,"patch_size":cfg.get("patch_size")}
