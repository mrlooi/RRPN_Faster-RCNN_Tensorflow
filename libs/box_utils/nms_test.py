import numpy as np

def DIVUP(m, n): 
    return m // n + ((m) % (n) > 0)

threadsPerBlock = 4

def fake_kernel(dev_boxes, diff_thresh):
    N = dev_boxes.shape[0]
    col_blocks = DIVUP(N, threadsPerBlock)
    mask_host = np.zeros((N, col_blocks), dtype=np.int64)
    
    for r in range(N):
        for cb in range(col_blocks):
            for tid in range(threadsPerBlock):
                idx = cb * threadsPerBlock + tid
                c = idx
                if c >= N:
                    break
                if c <= r:
                    continue
                diff = np.abs(dev_boxes[r] - dev_boxes[c])
                if diff < diff_thresh:
                    mask_host[r,cb] |= 1 << tid

        # for c in range(N):
        #     if r == c:
        #         continue
        #     diff = np.abs(dev_boxes[r] - dev_boxes[c])
        #     if diff < diff_thresh:
        #         i = c // threadsPerBlock
        #         j = c % threadsPerBlock
        #         mask_host[r,i] |= 1 << j
    return mask_host

N = 10
col_blocks = DIVUP(N, threadsPerBlock)

# x = np.arange(N)
# thresh = 2
x = np.random.random(size=N)
x = np.array([0.22,0.3,0.6,0.1,0.4,0.9,0.8,0.99,0.7,0.01])
thresh = 0.15

# x = 
mask_host = fake_kernel(x, diff_thresh=thresh)
print(mask_host)

remv = np.zeros(col_blocks, dtype=np.int64)
keep_out = np.zeros(N, dtype=np.uint64)

num_to_keep = 0
for i in range(N):
    nblock = i // threadsPerBlock
    inblock = i % threadsPerBlock

    if not (remv[nblock] & (1 << inblock)):
        keep_out[num_to_keep] = i
        num_to_keep += 1

        p = mask_host[i]
        for j in range(nblock, col_blocks):
            remv[j] |= p[j]

keep = keep_out[:num_to_keep]
print(keep)