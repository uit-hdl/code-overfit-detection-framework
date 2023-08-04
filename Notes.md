## Personal notes
To have 16-bit floating point precision (less accurate, less memory-intensive)
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output, target = model(im_q=images_q.cuda(), im_k=images_k.cuda())
```