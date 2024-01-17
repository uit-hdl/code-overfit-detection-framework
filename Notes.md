## Personal notes
To have 16-bit floating point precision (less accurate, less memory-intensive)
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output, target = model(im_q=images_q.cuda(), im_k=images_k.cuda())
```

## Datasets
I've tried cachedataset, persistentdataset and smartcachedataset. They all take the same amount of memory during a run.

Increasing the amount of data doesn't matter, suggesting that a single batch of 96 consumes 20 GB of memory.

## Sequential checkpoints
Trade memory for compute. Doesn't scale with distributed

## Azure
NVIDIA AI Enterprise - 30 k i mnd :p

Standard NC12s v3 has 2 GPU.

But it's 60 k per month
A single GPU NC6 costs 23 K per month

... But this is deprecated, migrate to Standard NC48ads A100 v4
But its not deprecated? Cant find it
NV12 looks ok.

### Container instances
Container instances may work, but they have at most 15 GB of storage....

[Pricing details](https://azure.microsoft.com/nb-no/pricing/details/container-instances/)

K80 GPUs are now deprecated

297 per vCPU
39 per GB

Price estimations
```bc
# K80 GPU
# price per month
3366*2+297*2+39*64
9822
# price per day
327

# V100 GPU
26709*2+297*2+39*64
56508
# price per day
1883
```

## From talk
clusters are subjectively assigned, which is a weakness

was it the cluster labels that was the explainable part, not the UMAPs?
