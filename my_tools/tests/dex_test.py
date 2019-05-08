import numpy as np

def DexX(bottom_rois, i_int, j_int, pooled_height_int, pooled_width_int):
    i = float(i_int);
    j = float(j_int);
    pooled_width = float(pooled_width_int);
    pooled_height = float(pooled_height_int);

    return (pooled_height - i) / pooled_height * (
            (pooled_width - j) / pooled_width * bottom_rois[1 - 1] + j / pooled_width * bottom_rois[
        3 - 1]) + i / pooled_height * (
                   (pooled_width - j) / pooled_width * bottom_rois[7 - 1] + j / pooled_width * bottom_rois[5 - 1])


def DexY(bottom_rois, i_int, j_int, pooled_height_int, pooled_width_int):
    i = float(i_int);
    j = float(j_int);
    pooled_width = float(pooled_width_int);
    pooled_height = float(pooled_height_int);

    return (pooled_width - j) / pooled_width * (
            (pooled_height - i) / pooled_height * bottom_rois[2 - 1] + i / pooled_height * bottom_rois[
        8 - 1]) + j / pooled_width * (
                   (pooled_height - i) / pooled_height * bottom_rois[4 - 1] + i / pooled_height * bottom_rois[6 - 1]);


def cross_mul(pt1, pt2, pt3):
    return pt2[0] * pt3[1] + pt3[0] * pt1[1] + pt1[0] * pt2[1] - pt2[0] * pt1[1] - pt3[0] * pt2[1] - pt1[0] * pt3[1]


def inpoly(pt_x, pt_y, pts):
    flag = True;
    cur_sign = 0;
    pt = [0, 0]
    pt[0] = pt_x;
    pt[1] = pt_y;
    sign = 0
    for i in range(4):
        val = cross_mul(pts[i * 2:], pts[((i + 1) % 4 * 2):], pt);

        if val < 0.0:
            cur_sign = -1;
        elif val > 0.0:
            cur_sign = 1;
        else:
            cur_sign = 0;

        if cur_sign != 0:
            if flag:
                flag = False;
                sign = cur_sign;
            else:
                if sign != cur_sign:
                    return False;
    return True;


def get_bounds_of_rect_pts(P, max_height, max_width):
    leftMost = max(min(min(P[0], P[2]), min(P[4], P[6])), 0.0)
    topMost = max(min(min(P[1], P[3]), min(P[5], P[7])), 0.0)
    rightMost = min(max(max(P[0], P[2]), max(P[4], P[6])), max_width - 1.0)
    bottomMost = min(max(max(P[1], P[3]), max(P[5], P[7])), max_height - 1.0)

    return [leftMost, topMost, rightMost, bottomMost]


def get_dex_pts(roi_pts, ph, pw, pooled_height, pooled_width, spatial_scale):
    P = np.zeros(8, dtype=np.float32)

    P[0] = DexX(roi_pts, ph, pw, pooled_height, pooled_width) * spatial_scale;
    P[1] = DexY(roi_pts, ph, pw, pooled_height, pooled_width) * spatial_scale;
    P[2] = DexX(roi_pts, ph, pw + 1, pooled_height, pooled_width) * spatial_scale;
    P[3] = DexY(roi_pts, ph, pw + 1, pooled_height, pooled_width) * spatial_scale;
    P[4] = DexX(roi_pts, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale;
    P[5] = DexY(roi_pts, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale;
    P[6] = DexX(roi_pts, ph + 1, pw, pooled_height, pooled_width) * spatial_scale;
    P[7] = DexY(roi_pts, ph + 1, pw, pooled_height, pooled_width) * spatial_scale;

    return P


def is_point_in_rbox(rbox_pts, rbox_line_params, x, y):
    num_pos_sign = 0;
    num_neg_sign = 0;

    all_s = 0.0
    for j in range(4):
        # line equation: Ax + By + C = 0
        # see which side of the line this point is at
        A = -rbox_line_params[j * 2 + 1];
        B = rbox_line_params[j * 2];
        C = -(A * rbox_pts[j * 2] + B * rbox_pts[j * 2 + 1]);
        s = A * x + B * y + C;
        # print(s)
        if s > 0:
            num_pos_sign += 1;
        else:
            num_neg_sign += 1;

    return (num_pos_sign == 4 or num_neg_sign == 4)
