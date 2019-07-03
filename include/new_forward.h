#pragma once

#ifndef _NEW_FORWARD_H_
#define _NEW_FORWARD_H_

#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string.h>
#include "opencv2/ml.hpp"
#include "new_forward.h"

using namespace std;
using namespace cv;
using namespace ml;

int const depth_pixel = 50;
int const pixel_error = 15;
int const black_thresh = 10;
int const white_thresh = 120;
int const white_thresh_paizi = 170;

int const image_cols = 20;
int const image_rows = 30;

int const ellipse_lowest = 30;
int const num_lowest = 50;
float const width_height_ratio = 1.5;
int const minArea = 200;

struct rect_params
{
	Point2f  center;
	float width;
	float height;
};

map<int, rect_params> get_forward_result(Mat src_img, Ptr<KNearest> num_knn);

#endif // !_NEW_FORWARD_H_
