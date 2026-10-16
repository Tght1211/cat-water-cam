"""可选S3D主干微调：只训练末段特征层，验证集选轮次；不接触独立测试集。"""
from pathlib import Path
import time
import numpy as np

from catcam.videojudge import read_temporal_windows


def fine_tune_s3d(clips_dir,names,y,train_indices,val_indices,models_dir,roi=None,epochs=8,progress_cb=None,paths=None):
    import torch
    from torchvision.models.video import s3d,S3D_Weights
    torch.manual_seed(0)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    weights=S3D_Weights.DEFAULT
    model=s3d(weights=weights)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.features[-3:].parameters():parameter.requires_grad_(True)
    model.classifier=torch.nn.Conv3d(1024,1,1)
    model.to(device)
    transform=weights.transforms()
    n_pos=max(1,sum(y[i]==1 for i in train_indices));n_neg=max(1,sum(y[i]==0 for i in train_indices))
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/n_pos],device=device))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    models_dir=Path(models_dir);models_dir.mkdir(parents=True,exist_ok=True)
    path=models_dir/f's3d_backbone_{time.time_ns()}.pt'
    best=float('inf');rng=np.random.default_rng(0)

    def sample(index,training):
        windows=read_temporal_windows(paths[names[index]] if paths else Path(clips_dir)/names[index])
        if not windows:raise ValueError(f'{names[index]}没有可读的原视频；主干微调需要原视频')
        # 连续窗口保留舔水时间尺度；避免同时把所有视频解码到内存。
        inputs=[]
        for frames in windows:
            if roi:
                x1,y1,x2,y2=roi;h,w=frames[0].shape[:2]
                frames=[f[int(max(0,y1-.1)*h):int(min(1,y2+.1)*h),int(max(0,x1-.1)*w):int(min(1,x2+.1)*w)] for f in frames]
            clip=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)
            inputs.append(transform(clip).unsqueeze(0).to(device))
        return inputs,torch.tensor([[float(y[index])]],device=device)

    def session_logits(inputs):
        # 会话标签只说明某个窗口存在喝水，不把靠近/离开窗口强行标成喝水。
        return torch.stack([model(window) for window in inputs]).amax(0)

    for epoch in range(epochs):
        # batch=1时冻结BN统计，保留预训练分布；仅末段卷积与临时分类头参与梯度。
        model.eval();train_loss=0
        if progress_cb:
            progress_cb({'phase':'finetuning','epoch':epoch,'epochs':epochs,'loss':0,'val_loss':best if np.isfinite(best) else 0,
                         'sample':0,'samples':len(train_indices)})
        for done,index in enumerate(rng.permutation(train_indices),1):
            inputs,target=sample(index,True)
            optimizer.zero_grad();loss=loss_fn(session_logits(inputs),target);loss.backward();optimizer.step()
            train_loss+=float(loss.detach().cpu())
            if progress_cb:
                progress_cb({'phase':'finetuning','epoch':epoch,'epochs':epochs,'loss':train_loss/done,
                             'val_loss':best if np.isfinite(best) else 0,'sample':done,'samples':len(train_indices)})
        val_loss=0
        with torch.inference_mode():
            for index in val_indices:
                inputs,target=sample(index,False)
                val_loss+=float(loss_fn(session_logits(inputs),target).cpu())
        val_loss/=len(val_indices)
        if val_loss<best:
            best=val_loss
            state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('classifier.')}
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
        if progress_cb:
            progress_cb({'phase':'finetuning','epoch':epoch+1,'epochs':epochs,
                         'loss':train_loss/len(train_indices),'val_loss':val_loss,'sample':0,'samples':len(train_indices)})
    return path
