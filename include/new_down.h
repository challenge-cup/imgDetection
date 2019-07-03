#pragma once

#ifndef _NEW_DOWN_H_
#define _NEW_DOWN_H_

#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string.h>
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace ml;


int const sample_num_perclass = 200;		//每类训练样本数量
int const class_num = 10;					//类别，数字加字母
int const image_cols_num = 20;
int const image_rows_num = 30;
int const lowest_num = 200;					//数字面板最小面积

struct image_params {
	Point2f center;
	Mat roi;
};


void getTreeTop(cv::Mat &_img, cv::Point &center, bool &state);


map<int, Point2f> get_down_result(Mat src_image, Ptr<KNearest> num_knn);
void distCorrect(cv::Mat &src, cv::Mat &dst);

#endif // !_NEW_DOWN_H_
