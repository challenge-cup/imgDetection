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

//提取椭圆
Mat image_init(Mat src_img)
{
	Mat gray_img, binary_img;
	// 灰度化
	cvtColor(src_img, gray_img, COLOR_BGR2GRAY);

	//blur(gray_img, gray_img, Size(2, 2));
	//imshow("【去噪后】", gray_img);

	// 二值化
	threshold(gray_img, binary_img, black_thresh, 255, THRESH_BINARY_INV);
	//imshow("二值化", binary_img);
	
	// 开闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(binary_img, binary_img, MORPH_OPEN, element);
	//morphologyEx(binary_img, binary_img, MORPH_CLOSE, element);
	//imshow("开闭操作", binary_img);

	return binary_img;
}

//提取白色区域
Mat image_init_new(Mat src_img)
{
	Mat gray_img, binary_img;
	// 灰度化
	cvtColor(src_img, gray_img, COLOR_BGR2GRAY);

	//blur(gray_img, gray_img, Size(2, 2));
	//imshow("【去噪后】", gray_img);

	// 二值化
	//threshold(gray_img, binary_img, white_thresh_paizi, 255, THRESH_BINARY);
	//imshow("二值化", binary_img);

	//canny边缘提取
	Canny(gray_img, binary_img, 80, 100);

	//Mat dst(binary_img.cols, binary_img.rows, CV_32FC1, Scalar(0));
	//src_img.copyTo(dst, binary_img)

	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	//膨胀腐蚀//膨胀操作
	dilate(binary_img, binary_img, element);

	// 开闭操作
	//morphologyEx(binary_img, binary_img, MORPH_OPEN, element);
	//morphologyEx(binary_img, binary_img, MORPH_CLOSE, element);
	//imshow("开闭操作", binary_img);
	//waitKey(1);

	return binary_img;
}

Mat image_init_hsv_new(Mat src_img)
{
	int iLowH = 14;
	int iHighH = 44;

	int iLowS = 86;
	int iHighS = 255;

	int iLowV = 42;
	int iHighV = 201;
	Mat imgHSV, imgThresholded;

	cvtColor(src_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
												//cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  

	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);

																								  //开闭运算
	Mat bin_ero;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(imgThresholded, bin_ero, element);
	//morphologyEx(binary_image, binary_image, MORPH_OPEN, element);
	//morphologyEx(binary_image, binary_image, MORPH_CLOSE, element);
	//imshow("【开闭运算】", binary_image);

	//取反
	//binary_image = 255 - binary_image;
	//imshow("new_down", binary_image);
	//waitKey(1);
	//cv::imshow("img", imgThresholded);
	//cv::waitKey(0);

	return imgThresholded;
}

Mat image_hsv(Mat src_img)
{
	//hsv
	Mat src_hsv;
	vector<Mat> hsvSplit;
	cvtColor(src_img, src_hsv, COLOR_BGR2HSV);

	//因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
	split(src_hsv, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, src_hsv);

	inRange(src_hsv, Scalar(0, 0, 190), Scalar(179, 255, 255), src_hsv);

	return src_hsv;
}

Rect findrect(Mat binary_img)
{
	Rect roi_rect;

	vector<vector<Point>> contours;		//寻找轮廓
	vector<Rect> white_rect;
	findContours(binary_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	double area_whole = 0.7;	//定义最小的连通域面积与矩形框面积之比
	int max_possible_index = -1;	//初始化最大可能方框的下标

	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundingRect(contours[i]);
		double area = contourArea(contours[i]);
		//矩形框面积不能过大，轮廓连通域面积不能过小，符合一定长宽比
		//if (rect.width*rect.height / (binary_img.cols*binary_img.rows) < 0.4&&area > minArea&&rect.width / (rect.height + 0.01) > 1.2&&rect.width / (rect.height + 0.01) < 2.1)
		if (rect.width*rect.height / (binary_img.cols*binary_img.rows) < 0.4&&area > minArea&&rect.width / (rect.height + 0.01) > 0.7&&rect.width / (rect.height + 0.01) < 1.3)
		{
			//if(binary_img.at<int>(rect.x+rect.width/2.0,rect.y+rect.height/2.0)>128)
			if (area / (rect.width*rect.height) > area_whole)
			{
				area_whole = area / (rect.width*rect.height);
				max_possible_index = i;
				//rectangle(binary_img, rect, Scalar(0), 1);
				//cout << contourArea(contours[i]) << " " << rect.width << " " << rect.height <<" "<< contourArea(contours[i]) / (rect.width*rect.height) << endl;
			}
		}
	}
	if (max_possible_index > -1)
	{
		roi_rect = boundingRect(contours[max_possible_index]);
		//Mat roi = binary_img(roi_rect);
		//imshow("roi", roi);
		//waitKey(0);
	}
	return roi_rect;
}

vector<RotatedRect> findellipse(Mat binary_img)
{
	int i, j;
	vector<vector<Point>> contours;		//寻找轮廓
	findContours(binary_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<RotatedRect> ellipse_temp;
	vector<RotatedRect> minEllipse;		//保存同心圆
	int ellipse_count = 0;				//同心圆组数

	for (i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > ellipse_lowest)
		{
			ellipse_temp.push_back(fitEllipse(Mat(contours[i])));
			Point2f vertices[4];
			ellipse_temp[ellipse_count].points(vertices);
			for (int k = 0; k < 4; k++)
				line(binary_img, vertices[k], vertices[(k + 1) % 4], Scalar(255), 1);

			for (j = 0; j != ellipse_temp.size(); j++)
			{
				if (fabsf(ellipse_temp[ellipse_count].center.x - ellipse_temp[j].center.x) < 3 && fabsf(ellipse_temp[ellipse_count].center.y - ellipse_temp[j].center.y) < 3)
					break;
			}
			if (j < ellipse_count)
			{
				minEllipse.push_back(ellipse_temp[ellipse_count]);
				minEllipse.push_back(ellipse_temp[j]);
			}

			ellipse_count++;

		}

	}
	return minEllipse;
}

Mat get_num_roi_almost_new(Rect roi_rect, Mat src_img)
{
	Mat num_roi_almost = src_img(roi_rect);
	//imshow("almost", num_roi_almost);
	//waitKey(1);
	return num_roi_almost;
}

Mat get_num_roi_almost(Mat src_img, Mat binary_img, RotatedRect Ellipse)
{
	//cout << Ellipse.size << endl;

	int num_x, num_y, num_height, num_width;
	int xy_range = Ellipse.size.height > Ellipse.size.width ? Ellipse.size.height : Ellipse.size.width;
	num_x = Ellipse.center.x - xy_range / 2;
	num_y = Ellipse.center.y + xy_range / 2;
	num_width = num_x + xy_range < binary_img.cols ? xy_range : binary_img.cols - num_x;
	num_height = num_y + xy_range / width_height_ratio < binary_img.rows ? xy_range / width_height_ratio : binary_img.rows - num_y;
	Mat num_roi_almost = src_img(Rect(num_x, num_y, num_width, num_height));
	//imshow("数字大概区域", num_roi_almost);
	//waitKey();
	return num_roi_almost;
}

Mat hsv_process(Mat num_roi_almost)
{
	Mat num_roi_hsv;
	vector<Mat> hsvSplit;
	cvtColor(num_roi_almost, num_roi_hsv, COLOR_BGR2HSV);

	//因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
	split(num_roi_hsv, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, num_roi_hsv);

	inRange(num_roi_hsv, Scalar(0, 0, 190), Scalar(179, 255, 255), num_roi_hsv);
	//imshow("hsv", num_roi_hsv);
	//waitKey();
	return num_roi_hsv;
}

Mat num_roi_process(Mat num_roi_almost)
{
	//将数字的roi区域二值化，切除多余部分
	//二值化
	//cvtColor(num_roi_almost, num_roi_almost, COLOR_BGR2GRAY);
	//threshold(num_roi_almost, num_roi_almost, white_thresh, 255, THRESH_BINARY);

	//开闭操作
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(num_roi_almost, num_roi_almost, MORPH_OPEN, element);
	//morphologyEx(num_roi_almost, num_roi_almost, MORPH_CLOSE, element);

	//寻找数字轮廓
	vector<vector<Point>> num_contours;
	findContours(num_roi_almost, num_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat Number;
	for (int i = 0; i < num_contours.size(); i++) {
		Rect rect = boundingRect(num_contours[i]);
		if (1.0 * rect.width / rect.height<1.0 && 1.0 * rect.width / rect.height>0.3 && rect.width*rect.height>num_lowest&&(1.0*rect.width*rect.height / (num_roi_almost.cols*num_roi_almost.rows)<0.5))
		{
			//cout << 1.0*rect.width*rect.height / (num_roi_almost.cols*num_roi_almost.rows) << endl;
			//cout << rect.width << " " << rect.height << endl;
			//rectangle(num_roi, rect, Scalar(0), 1);			// 绘制数字矩形
			Number = num_roi_almost(rect);
			//imshow("forward轮廓", Number);
			//waitKey(1);
			break;
		}
		//if (1.0 * rect.width / rect.height<1.2 || 1.0 * rect.width / rect.height>2.5 || rect.height * rect.width<200)
		//	continue;

		//Mat roi = num_roi(rect);
		//if (1.0 * countNonZero(roi) / (roi.rows*roi.cols) < 0.5)
		//	continue;

		//Number.insert(pair<int, Mat>(rect.x, roi));
	}
	return Number;
}

int num_predict(Mat num_roi, Ptr<KNearest> num_knn)
{
	int K = 3;	// num_knn->getDefaultK;
	Mat img_temp;

	resize(num_roi, img_temp, Size(image_cols, image_rows), (0, 0), (0, 0), INTER_AREA);//统一图像
	threshold(img_temp, img_temp, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("【roi】", img_temp);
	Mat sample_mat(1, image_cols*image_rows, CV_32FC1);
	for (int i = 0; i < image_rows*image_cols; ++i)
		sample_mat.at<float>(0, i) = (float)img_temp.at<uchar>(i / image_cols, i % image_cols);
	Mat matResults;//保存测试结果
	num_knn->findNearest(sample_mat, K, matResults);//knn分类预测    
	//cout << matResults << endl;
	return (int)matResults.at<float>(0, 0);
}

map<int, rect_params> get_forward_result(Mat src_img, Ptr<KNearest> num_knn)
{
	//map<int, ellipse_params> num_ellipse;
	map<int, rect_params> num_rect;

	Mat binary_img = image_init_hsv_new(src_img).clone();
	cv::imshow("binary_img", binary_img);
	cv::waitKey(0);

	Rect roi_rect = findrect(binary_img);
	if (!roi_rect.empty())
	{
		Mat num_roi_almost = get_num_roi_almost_new(roi_rect, binary_img);
		Mat Number = num_roi_process(num_roi_almost).clone();
		if (!Number.empty())
		{
			//cout << "matResults=" << num_predict(Number, num_knn) << endl;
			rect_params rp;
			rp.center = Point2f(roi_rect.x + roi_rect.width / 2, roi_rect.y + roi_rect.height / 2);
			rp.width = roi_rect.width;
			rp.height = roi_rect.height;
			num_rect.insert(pair<int, rect_params>(num_predict(Number, num_knn), rp));
		}
	}

	//Mat src_hsv = image_hsv(src_img).clone();

	/*
	Mat binary_img = image_init(src_img).clone();

	vector<RotatedRect> minEllipse = findellipse(binary_img);

	//截取同心圆下方的数字方块
	for (int i = 0; i < minEllipse.size() / 2; i++)
	{
		RotatedRect max_ellipse, min_ellipse;	//同心圆中的大圆与小圆
		if (minEllipse[2 * i].size.height > minEllipse[2 * i + 1].size.height)
		{
			max_ellipse = minEllipse[2 * i];
			min_ellipse = minEllipse[2 * i + 1];
		}
		else
		{
			min_ellipse = minEllipse[2 * i];
			max_ellipse = minEllipse[2 * i + 1];
		}

		Mat num_roi_almost = get_num_roi_almost(src_img, binary_img, min_ellipse).clone();

		struct ellipse_params ep;
		//min_ellipse.center,
		//min_ellipse.size.width,
		//min_ellipse.size.height
		ep.center = min_ellipse.center;
		if (fabsf(min_ellipse.angle) > 45)
		{
			ep.height = min_ellipse.size.width;
			ep.width = min_ellipse.size.height;
		}
		else
		{
			ep.width = min_ellipse.size.width;
			ep.height = min_ellipse.size.height;
		}

		//hsv提取数字区域
		//Mat num_roi_hsv = hsv_process(num_roi_almost).clone();

		Mat Number = num_roi_process(num_roi_almost).clone();

		if (!Number.empty())
		{
			cout << "matResults=" << num_predict(Number, num_knn) << endl;

			num_ellipse.insert(pair<int, ellipse_params>(num_predict(Number, num_knn), ep));
		}
	}*/

	//imshow("binary_img", binary_img);
	//waitKey();

	return num_rect;
}