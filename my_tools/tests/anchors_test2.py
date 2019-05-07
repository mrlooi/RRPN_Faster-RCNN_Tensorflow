import cv2
import numpy as np

RED = [0,0,255]
BLUE = [255,0,0]
GREEN = [0,255,0]

if __name__ == '__main__':
    from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

    H, W = (100, 100)
    mh = H // 2
    mw = W // 2
    rect_pairs = np.array([
        # [[mh,mw,15,30,-30], [mh,mw,30,15,-90]],
        # [[mh,mw,15,30,-90], [mh,mw,10,45,-90]],
        # [[mh,mw,15,10,-90], [mh,mw,20,5,-90]],
        # [[mh,mw,15,10,-90], [mh,mw,10,20,-90]],
        # [[mh,mw,10,10,-45], [mh,mw,20,12,-45]],

        # [[mh,mw,10,15,-90], [mh,mw,20,5,-45]],
        # [[mh,mw,10,15,-90], [mh,mw,20,5,-60]],
        # [[mh,mw,10,15,-90], [mh,mw,20,5,-20]],
        # [[mh,mw,20,10,-60], [mh,mw,5,10,-60]],
        # [[mh,mw,20,10,-60], [mh,mw,10,5,-60]],
        [[mh,mw,15,28,-86], [mh,mw,90,45,-30]],
        [[mh,mw,15,28,-86], [mh,mw,45,90,-30]],
        [[mh,mw,15,28,-86], [mh,mw,90,45,-60]],
        [[mh,mw,15,28,-86], [mh,mw,45,90,-60]],
        [[mh,mw,37,16,-86], [mh,mw,90,45,-30]],
        [[mh,mw,37,16,-86], [mh,mw,45,90,-30]],
        [[mh,mw,37,16,-86], [mh,mw,90,45,-60]],
        [[mh,mw,37,16,-86], [mh,mw,45,90,-60]],
    ], dtype=np.float32) 
    rect_pairs = rect_pairs#[-2:]
    
    VIS_SCALE = 4
    for r_pairs in rect_pairs:

        r1 = r_pairs[0].copy()  # target
        r2 = r_pairs[1].copy()  # anchor

        w1,h1,angle1 = r1[2:]
        w2,h2,angle2 = r2[2:]

        r3 = r1.copy()
        if np.abs(angle1-angle2) > 45:
            r3[-1] -= np.sign(angle1-angle2) * 90
            r3[2:4] = r1[2:4][::-1].copy()
        print(r3, r1)

        # rr1 = convert_pts_to_rect(convert_rect_to_pts(r1), make_width_larger=False)
        # rr2 = convert_pts_to_rect(convert_rect_to_pts(r2), make_width_larger=False)
        # print(rr1, rr2)

        r1[:-1] *= VIS_SCALE
        r2[:-1] *= VIS_SCALE
        r3[:-1] *= VIS_SCALE

        img = np.zeros((H*VIS_SCALE, W*VIS_SCALE,3), dtype=np.uint8)
        img = draw_anchors(img, [r1], [RED])
        img = draw_anchors(img, [r2], [BLUE])
        img = draw_anchors(img, [r3], [GREEN])
        cv2.imshow("img", img)
        cv2.waitKey(0)
