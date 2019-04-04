import numpy as np

def bilinear_interpolate(bottom_data, height, width, y, x):
    if (y < -1.0 or y > height or x < -1.0 or x > width):
        val = w1 = w2 = w3 = w4 = 0.0
        x_low = x_high = y_low = y_high = -1.0
        return [val, w1, w2, w3, w4, x_low, x_high, y_low, y_high]

    if y <= 0:
        y = 0
    if x <= 0:
        x = 0

    y_low = int(y)
    x_low = int(x)

    if (y_low >= height - 1):
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1

    if (x_low >= width - 1):
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx
    # do bilinear interpolation
    v1 = bottom_data[y_low * width + x_low]
    v2 = bottom_data[y_low * width + x_high]
    v3 = bottom_data[y_high * width + x_low]
    v4 = bottom_data[y_high * width + x_high]
    w1, w2, w3, w4 = hy * hx, hy * lx, ly * hx, ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return [val, w1, w2, w3, w4, x_low, x_high, y_low, y_high]

class RoiAlignCpu(object):
    def __init__(self, pool_size, spatial_scale=1.0, sampling_ratio=0):
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.pooled_height, self.pooled_width = pool_size

        self.bottom_data = None
        self.bottom_rois = None
        self.top_data = None

    def forward(self, image, rois):
        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        sampling_ratio = self.sampling_ratio
        spatial_scale = self.spatial_scale

        B, C, H, W = image.shape
        N = len(rois)

        top_data = np.zeros((N, C, pooled_height, pooled_width), dtype=image.dtype)

        for n in range(N):
            roi = rois[n]  # batch_ind, start_x, start_y, end_x, end_y

            batch_ind = int(roi[0])

            roi_start_w = roi[1] * spatial_scale
            roi_start_h = roi[2] * spatial_scale
            roi_end_w = roi[3] * spatial_scale
            roi_end_h = roi[4] * spatial_scale

            # Force malformed ROIs to be 1x1
            roi_width = max(roi_end_w - roi_start_w, 1.)
            roi_height = max(roi_end_h - roi_start_h, 1.)
            bin_size_h = float(roi_height) / pooled_height
            bin_size_w = float(roi_width) / pooled_width

            roi_bin_grid_h = int(sampling_ratio if sampling_ratio > 0 else np.ceil(bin_size_h))
            roi_bin_grid_w = int(sampling_ratio if sampling_ratio > 0 else np.ceil(bin_size_w))

            count = roi_bin_grid_h * roi_bin_grid_w

            for c in range(C):
                offset_bottom_data = image[batch_ind,c].flatten()
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        # index = n * C * pooled_height * pooled_width + c * pooled_height * pooled_width + \
                        #         ph * pooled_width + pw

                        output_val = 0.0
                        for iy in range(roi_bin_grid_h):
                            y = roi_start_h + ph * bin_size_h + (iy + .5) * bin_size_h / roi_bin_grid_h
                            for ix in range(roi_bin_grid_w):
                                x = roi_start_w + pw * bin_size_w + (ix + .5) * bin_size_w / roi_bin_grid_w
                                values = bilinear_interpolate(offset_bottom_data, H, W, y, x)
                                output_val += values[0]
                        output_val /= count

                        top_data[n, c, ph, pw] = output_val

        self.top_data = top_data.copy()
        self.bottom_data = image.copy()
        self.bottom_rois = rois.copy()

        return top_data

    def backward(self):
        assert self.top_data is not None and self.bottom_data is not None and self.bottom_rois is not None

        top_diff = np.ones_like(self.top_data) # assume gradient is just 1.0
        bottom_diff = np.zeros_like(self.bottom_data)

        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        sampling_ratio = self.sampling_ratio
        spatial_scale = self.spatial_scale

        B, C, H, W = self.bottom_data.shape
        rois = self.bottom_rois
        N = len(rois)

        for n in range(N):
            roi = rois[n]  # batch_ind, start_x, start_y, end_x, end_y

            batch_ind = int(roi[0])

            roi_start_w = roi[1] * spatial_scale
            roi_start_h = roi[2] * spatial_scale
            roi_end_w = roi[3] * spatial_scale
            roi_end_h = roi[4] * spatial_scale

            # Force malformed ROIs to be 1x1
            roi_width = max(roi_end_w - roi_start_w, 1.)
            roi_height = max(roi_end_h - roi_start_h, 1.)
            bin_size_h = float(roi_height) / pooled_height
            bin_size_w = float(roi_width) / pooled_width

            roi_bin_grid_h = int(sampling_ratio if sampling_ratio > 0 else np.ceil(bin_size_h))
            roi_bin_grid_w = int(sampling_ratio if sampling_ratio > 0 else np.ceil(bin_size_w))

            count = roi_bin_grid_h * roi_bin_grid_w

            for c in range(C):
                offset_bottom_data = self.bottom_data[batch_ind,c].flatten()
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        top_diff_this_bin = top_diff[n,c,ph,pw]

                        for iy in range(roi_bin_grid_h):
                            y = roi_start_h + ph * bin_size_h + (iy + .5) * bin_size_h / roi_bin_grid_h
                            for ix in range(roi_bin_grid_w):
                                x = roi_start_w + pw * bin_size_w + (ix + .5) * bin_size_w / roi_bin_grid_w
                                values = bilinear_interpolate(offset_bottom_data, H, W, y, x)
                                _, w1, w2, w3, w4, x_low, x_high, y_low, y_high = values

                                if x_low >= 0 and x_high >= 0 and y_low >= 0 and y_high >= 0:
                                    g1 = top_diff_this_bin * w1 / count
                                    g2 = top_diff_this_bin * w2 / count
                                    g3 = top_diff_this_bin * w3 / count
                                    g4 = top_diff_this_bin * w4 / count
                                    bottom_diff[batch_ind, c, y_low, x_low] += g1
                                    bottom_diff[batch_ind, c, y_low, x_high] += g2
                                    bottom_diff[batch_ind, c, y_high, x_low] += g3
                                    bottom_diff[batch_ind, c, y_high, x_high] += g4

        return bottom_diff

if __name__ == '__main__':
    # pool_size = (2, 2)
    pool_size = (1, 1) # Ph, Pw
    spatial_scale = 1.0
    sampling_ratio = 0

    N = 1
    C = 1
    W = 3
    H = 3

    im_x = np.arange(N * C * H * W).reshape((N, C, H, W)).astype(np.float32)
    rois = np.array([
        [0, 0.5, 0.5, W-0.5, H-0.5]  # batch_ind, start_x, start_y, end_x, end_y
    ], dtype=np.float32)

    roi_align_cpu = RoiAlignCpu(pool_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    out_cpu = roi_align_cpu.forward(im_x, rois)
    im_x_grad = roi_align_cpu.backward()

    import torch
    from layers.roi_pool import ROIPool
    from layers.roi_align import ROIAlign
    pooler = ROIPool(pool_size, spatial_scale=spatial_scale)
    pooler2 = ROIAlign(pool_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)

    tx = torch.tensor(im_x, requires_grad=True, device='cuda')
    tx2 = torch.tensor(im_x, requires_grad=True, device='cuda')

    trois = torch.tensor(rois, device='cuda')

    out = pooler(tx, trois)
    loss = out.sum()
    loss.backward()

    out2 = pooler2(tx2, trois)
    loss2 = out2.sum()
    loss2.backward()

    print(out)
    print(out2)

