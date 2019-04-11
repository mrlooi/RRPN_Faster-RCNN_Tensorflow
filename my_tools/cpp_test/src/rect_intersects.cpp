#include <iostream>
#include <opencv2/opencv.hpp>

#include "rotate_rect_ops.h"

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

using namespace cv;
using std::max;
using std::min;


template <typename T>
inline void convert_region(T * pts , const RotatedRect& roi)
{
    T cx = roi.center.x;
    T cy = roi.center.y;
    T w = roi.size.width;
    T h = roi.size.height;
    T angle = deg2rad(roi.angle);

    T b = cos(angle)*0.5f;
    T a = sin(angle)*0.5f;

    pts[0] = cx - a*h - b*w;
    pts[1] = cy + b*h - a*w;
    pts[2] = cx + a*h - b*w;
    pts[3] = cy - b*h - a*w;
    pts[4] = 2*cx - pts[0];
    pts[5] = 2*cy - pts[1];
    pts[6] = 2*cx - pts[2];
    pts[7] = 2*cy - pts[3];
}


void draw_rect(cv::Mat& image, const RotatedRect& rRect)
{
    // rect vertices
    Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
    
    // bounding box of rect
    Rect brect = rRect.boundingRect();
    rectangle(image, brect, Scalar(255,0,0));
    
    imshow("rectangles", image);
    waitKey(0);
}

int get_rid_of_dupes(std::vector<Point2f>& intersection, const float samePointEps)
{
    // Get rid of dupes
    for( int i = 0; i < (int)intersection.size()-1; i++ )
    {
        for( size_t j = i+1; j < intersection.size(); j++ )
        {
            float dx = intersection[i].x - intersection[j].x;
            float dy = intersection[i].y - intersection[j].y;
            double d2 = dx*dx + dy*dy; // can be a really small number, need double here

            if( d2 < samePointEps*samePointEps )
            {
                // Found a dupe, remove it
                std::swap(intersection[j], intersection.back());
                intersection.pop_back();
                j--; // restart check
            }
        }
    }
    return intersection.size();
}


bool compare2(cv::Point2f a, cv::Point2f b)
{
    return a.x<b.x || (a.x==b.x && a.y<b.y);
}
//Returns positive value if B lies to the left of OA, negative if B lies to the right of OA, 0 if collinear
template <typename Point>
inline double cross(const Point &O, const Point &A, const Point &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

//Returns a list of points on the convex hull
template <typename Point>
std::vector<Point> convexHull(std::vector<Point> P)
{
    // taken from https://www.hackerearth.com/practice/math/geometry/line-sweep-technique/tutorial/
    int n = P.size(), k = 0;
    std::vector<Point> H(2*n);
    std::sort(P.begin(), P.end(), compare2);
    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++] = P[i];
    }

    // Build upper hull
    //i starts from n-2 because n-1 is the point which both hulls will have in common
    //t=k+1 so that the upper hull has atleast two points to begin with
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++] = P[i];
    }
    //the last point of upper hull is same with the fist point of the lower hull
    H.resize(k-1);
    return H;
}

// template <typename T>
void rotated_rect_pixel_interpolation(const RotatedRect& rect, const cv::Mat& image, const float spatial_scale=1.0f)
{
    int W = image.cols;
    int H = image.rows;

    float rect_vertices[8];
    float _roi_f[6] = {0, rect.center.x, rect.center.y, rect.size.width, rect.size.height, rect.angle};    
    compute_roi_pool_pts(_roi_f, rect_vertices, spatial_scale, 1, 1, 0, 0);

    // convert_region(rect_vertices, rect);

    // bounding box of rect
    float* P = rect_vertices;
    int leftMost = int(max(min(min(P[0], P[2]), min(P[4], P[6])), 0.0f));
    int topMost = int(max(min(min(P[1], P[3]), min(P[5], P[7])), 0.0f));
    int rightMost = int(min(max(max(P[0], P[2]), max(P[4], P[6])) + 1, W - 1.0f));
    int bottomMost = int(min(max(max(P[1], P[3]), max(P[5], P[7])) + 1, H - 1.0f));

    PRINT(leftMost)
    PRINT(topMost)
    PRINT(rightMost)
    PRINT(bottomMost)

    float roi_area = rect.size.height * spatial_scale * rect.size.width * spatial_scale;

    cv::Mat weights = cv::Mat::zeros(bottomMost - topMost + 1, rightMost - leftMost + 1, CV_32F);
    float output_val = 0.f;
    for(int hh = topMost; hh < bottomMost + 1; hh++)
    {
        for(int ww = leftMost; ww < rightMost + 1; ww++)
        {
            // RotatedRect pixel_rect = RotatedRect({ww+0.5f, hh+0.5f}, {1, 1}, 0); 
            // convert_region(pixel_rect_vertices, pixel_rect);
            float pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};

            float interArea = computeRectInterArea(rect_vertices, pixel_rect_vertices);
            if (interArea > 0)
            {
                printf("ww,hh: (%d,%d), inter_area: %.3f\n", ww,hh,interArea);

                float weight = interArea / roi_area;
                weights.at<float>(hh,ww) = weight;

                output_val += weight * image.at<float>(hh,ww);
            }
        }
    }
    PRINT(output_val)
}

void run_rotated_rect_pixel_interpolation()
{
    // Mat image(200, 200, CV_8UC3, Scalar(0));
    int H = 10;
    int W = 10;

    float spatial_scale = 0.8f;
    int PH = 1;
    int PW = 1;
    // std::vector<int> pool_dims {PH, PW}; 

    // rect 
    // cv::Point2f roi_center {3,3};  // xc,yc
    // cv::Point2f roi_size {3,3}; // w,h
    cv::Point2f roi_center {3,3};  // xc,yc
    cv::Point2f roi_size {3,3}; // w,h
    float roi_angle = 30;
    RotatedRect roi = RotatedRect(roi_center, roi_size, roi_angle);  // xc,yc,w,h,angle

    Point2f vertices[4];
    roi.points(vertices);
    for (int i = 0; i < 4; i++)
        printf("%.2f, %.2f, ", vertices[i].x, vertices[i].y); 
    printf("\n");

    float vertices_f[8];
    float _roi_f[6] = {0, roi_center.x, roi_center.y, roi_size.x, roi_size.y, roi_angle};    
    compute_roi_pool_pts(_roi_f, vertices_f, spatial_scale, PH, PW, 0, 0);
    for (int i = 0; i < 8; i++)
        printf("%.2f, ", vertices_f[i]); 
    printf("\n");

    float roi_f[5] = {roi_center.x, roi_center.y, roi_size.x, roi_size.y, roi_angle};    
    convert_region_to_pts(roi_f, vertices_f);
    for (int i = 0; i < 8; i++)
        printf("%.2f, ", vertices_f[i]); 
    printf("\n");

    cv::Mat image = cv::Mat::zeros(H, W, CV_32F);
    for(size_t i = 0; i < H*W; i++)
    {
        image.at<float>(i) = i;
    }

    rotated_rect_pixel_interpolation(roi, image, spatial_scale);
}

void run_rotated_rect_iou()
{
    float roi1[5] = {50, 50, 100, 300, 0.};
    float roi2[5] = {50, 50, 100, 300, 0.};
    float iou = computeRectIoU(roi1, roi2);

    printf("IOU: %.3f\n", iou);
}

int main()
{
    run_rotated_rect_pixel_interpolation();
    run_rotated_rect_iou();

    return 0;
}