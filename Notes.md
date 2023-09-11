## Personal notes
To have 16-bit floating point precision (less accurate, less memory-intensive)
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output, target = model(im_q=images_q.cuda(), im_k=images_k.cuda())
```

## Datasets
I've tried cachedataset, persistentdataset and smartcachedataset. They all take the same amount of memory during a run.

Increasing the amount of data doesn't matter, suggesting that a single batch of 96 consumes 20 GB of memory.
