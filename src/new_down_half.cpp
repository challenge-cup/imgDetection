#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string.h>
#include "opencv2/ml.hpp"
#include "new_down.h"

using namespace std;
using namespace cv;
using namespace ml;


Mat down_image_init(Mat src_image)
{
	Mat gray_image, binary_image;

	//灰度化
	cvtColor(src_image, gray_image, COLOR_BGR2GRAY);

	//去噪
	//blur(gray_image, gray_image, Size(4, 4));
	//imshow("【去噪后】", gray_image);

	//sobel滤波
	/*
	Mat dx, dy;
	Sobel(gray_image, dx, CV_8U, 1, 0, 3, 1, 0);
	convertScaleAbs(dx, dx);
	Sobel(gray_image, dy, CV_8U, 0, 1, 3, 1, 0);
	convertScaleAbs(dy, dy);
	addWeighted(dx, 1.0, dy, 1.0, 0, gray_image);
	imshow("【sobel滤波】", gray_image);
	*/

	//二值化
	threshold(gray_image, binary_image, 120, 255, THRESH_BINARY);

	//开闭运算
	Mat bin_ero;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(binary_image, bin_ero, element);
	//morphologyEx(binary_image, binary_image, MORPH_OPEN, element);
	//morphologyEx(binary_image, binary_image, MORPH_CLOSE, element);
	//imshow("【开闭运算】", binary_image);

	//取反
	//binary_image = 255 - binary_image;
	//imshow("new_down", binary_image);
	//waitKey(1);
	return binary_image;
}

Mat down_image_hsv_init(Mat src_image)
{
	int iLowH = 26;
	int iHighH = 45;

	int iLowS = 70;
	int iHighS = 255;

	int iLowV = 46;
	int iHighV = 255;
	Mat imgHSV, imgThresholded;

	cvtColor(src_image, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
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

void fullScreenNum(cv::Mat &_img, cv::Mat &dst, std::vector<cv::Point2f> &vertex)
{

	//Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[])  ;
	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(_img.cols - 1, 0);
	corners[2] = Point2f(_img.cols - 1, _img.rows - 1);
	corners[3] = Point2f(0, _img.rows - 1);
	vector<Point2f> corners_trans(4);
	for (int i = 0; i <= 3; i++)
	{
		corners_trans[i] = vertex[i];
		//cout << "circleCenter:" << i << "=" << "(" << corners_trans[i].x << "," << corners_trans[i].y << ")" << endl;

	}
	Mat M = getPerspectiveTransform(corners_trans, corners);

	warpPerspective(_img, dst, M, Size(0, 0));

	//imshow("3", dst);

	//waitKey(0);	

}

map<int, image_params> image_split(Mat binary_image)
{
	//cv::imshow("binary_image", binary_image);
	//cv::waitKey(0);

	Mat binary_image_ref = binary_image.clone();
	Size image_size(binary_image.size());

	vector<vector<Point>> contours;
	findContours(binary_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	map<int, image_params> num_roi_almost;
	for (int i = 0; i < contours.size(); i++) {
		//drawContours(binary_image, contours, i, Scalar(255), 1); // 绘制轮廓

		RotatedRect rect = minAreaRect(contours[i]);

		if (rect.size.height / (rect.size.width + 0.01)<0.7 || rect.size.height / (rect.size.width + 0.01)>1.3 || rect.size.height * rect.size.width<lowest_num)
			continue;
		//if(rect.center.x-rect.size.width/2<0.01|| rect.center.x + rect.size.width / 2>binary_image.cols-0.01|| rect.center.y - rect.size.width / 2<0.01 || rect.center.x + rect.size.width / 2>binary_image.cols - 0.01)
		//删除边落在图像边界上的矩形
		//Point2f vertices[4];
		//rect.points(vertices);
		//int flag = 0;
		//for (int j = 0; j < 4; j++) {
		//	//	line(binary_image, vertices[j], vertices[(j + 1) % 4], Scalar(255), 1);
		//	if (vertices[j].x<0.01 || vertices[j].y<0.01 || vertices[j].x>binary_image.cols - 0.01 || vertices[j].y>binary_image.rows - 0.01)
		//	{
		//		flag = 1;
		//		break;
		//	}
		//}
		//if (flag == 1)
		//{
		//	flag = 0;
		//	continue;
		//}
		//cout << "h:" << rect.size.height << " w:" << rect.size.width << endl;
		//将旋转矩形转化成正矩形
		Mat rotation = getRotationMatrix2D(rect.center, rect.angle, 1.0);
		Mat rot_img;
		warpAffine(binary_image_ref, rot_img, rotation, image_size);

		
		Point corner[4] = { Point(0,0),
			Point(binary_image.cols - 1,0),
			Point(binary_image.cols - 1,binary_image.rows - 1),
			Point(0,binary_image.rows - 1)
			};

		Point2f vertices[4];
		rect.points(vertices);
		std::vector<cv::Point2f> vertex;

		for (int i = 0; i <= 3; i++)
		{
			int min = INT_MAX, minloc = 0;

			for (int j = i; j <= 3; j++)
			{
				int squareDistance = pow(vertices[j].x - corner[i].x, 2) + pow(vertices[j].y - corner[i].y, 2);

				if (squareDistance < min)
				{
					min = squareDistance;
					minloc = j;
				}
				//cout << squareDistance << "squareDistance" << endl;
				//cout << min << "min" << endl;
			}

			swap(vertices[i], vertices[minloc]);
		}
		for (int i = 0; i <= 3; i++)
		{
			vertex.push_back(vertices[i]);
		}

		Mat dst;
		fullScreenNum(binary_image, rot_img, vertex);

		//imshow("【binary_image】", binary_image);
		//imshow("【旋转】", rot_img);
		//waitKey(0);

		//cout << rect.center.x << " " << rect.center.y << " " << rect.size.width << " " << rect.size.height << endl;

		int imgx = rect.center.x - (rect.size.width / 2.0) > 0 ? (int)(rect.center.x - (rect.size.width / 2.0)) : 0;
		int imgy = rect.center.y - (rect.size.height / 2.0) > 0 ? (int)(rect.center.y - (rect.size.height / 2.0)) : 0;
		int imgwidth = (imgx + rect.size.width) < rot_img.cols ? (int)(rect.size.width) : (int)(rot_img.cols - imgx);
		int imgheight = (imgy + rect.size.height) < rot_img.rows ? (int)(rect.size.height) : (int)(rot_img.rows - imgy);
		Mat roi = rot_img(Rect(imgx, imgy, imgwidth, imgheight));
		//将图像放正
		//先把长边为宽，短边为高
		//if (roi.cols < roi.rows)
		//{
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//}

		//float x = 0.0f;		// roi.cols / 2.0;
		float y = 0.0f;		// roi.rows / 2.0;
		vector<vector<Point>> num_contours;
		findContours(roi, num_contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		//原来取偏差最大的，现在选择取平均，可以解决10的问题
		//vector<Point2f> point_temp;
		for (int k = 0; k < num_contours.size(); k++)
		{
			Rect num_rect = boundingRect(num_contours[k]);
			//if (num_rect.x<2 || num_rect.y<2 || num_rect.x + num_rect.width>roi.cols - 2 || num_rect.y + num_rect.height>roi.rows - 2)
			//	break;
			int flag = 0;

			//x += num_rect.x + num_rect.width / 2.0f;
			y += num_rect.y + num_rect.height / 2.0f;
			/*
			if (fabsf(num_rect.x + num_rect.width / 2.0 - roi.cols / 2.0) > fabsf(x - roi.cols / 2.0))
			x = num_rect.x + num_rect.width / 2.0;
			if (fabsf(num_rect.y + num_rect.height / 2.0 - roi.rows / 2.0) > fabsf(y - roi.rows / 2.0))
			y = num_rect.y + num_rect.height / 2.0;*/

			//rectangle(roi, num_rect, Scalar(0,0,255), 1);
		}
		//x /= num_contours.size();
		y /= num_contours.size();
		//cout << x << " " << roi.cols << " " << y << " " << roi.rows << endl;

		//if (y < roi.rows / 2.0)
		//{
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//}
		/*if (fabsf(x - roi.cols / 2.0) > fabsf(y - roi.rows / 2.0))
		{
		if (x > roi.cols / 2.0)
		{
		transpose(roi, roi);
		flip(roi, roi, 1);
		}
		else
		{
		transpose(roi, roi);
		flip(roi, roi, 0);
		}
		}
		else
		{
		if (y < roi.rows / 2.0)
		{
		transpose(roi, roi);
		flip(roi, roi, 1);
		transpose(roi, roi);
		flip(roi, roi, 1);
		}
		}*/

		struct image_params ip {
			rect.center,
				roi
		};

		num_roi_almost.insert(pair<int, image_params>(fabsf(rect.center.x - roi.cols), ip));
	}
	//imshow("【分割图像】", binary_image);
	//waitKey(1);
	return num_roi_almost;
}

Mat down_num_roi_process(Mat num_roi_almost)
{
	//寻找数字轮廓
	vector<vector<Point>> num_contours;
	findContours(num_roi_almost, num_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat Number;
	Mat temp;
	//int j = 0;

	for (int i = 0; i < num_contours.size(); i++) {
		Rect rect = boundingRect(num_contours[i]);
		if (1.0 * rect.width / (rect.height + 0.01)<0.9 && 1.0 * rect.width / (rect.height + 0.01)>0.3 && rect.width*rect.height>100)
		{
			//cout << rect.width << " " << rect.height << endl;
			//rectangle(num_roi, rect, Scalar(0), 1);			// 绘制数字矩形
			Number = num_roi_almost(rect);
			//j++;
			//imshow("down数字轮廓", Number);
			//waitKey(1);
			break;
		}
	}
	//if (j > 1) 
	//	return temp;
	//else 
	return Number;
	//j = 0;
}

//识别
int down_num_predict(Mat num_rect, Ptr<KNearest> num_knn)
{
	const int K = 3;	//testModel->getDefaultK()  
	Mat img_temp;
	resize(num_rect, img_temp, Size(image_cols_num, image_rows_num), (0, 0), (0, 0), INTER_AREA);//统一图像
	threshold(img_temp, img_temp, 0, 255, THRESH_BINARY | THRESH_OTSU);

	//imshow("【roi】", img_temp);
	//waitKey();

	Mat sample_mat(1, image_cols_num*image_rows_num, CV_32FC1);
	for (int i = 0; i < image_cols_num*image_rows_num; ++i)
		sample_mat.at<float>(0, i) = (float)img_temp.at<uchar>(i / image_cols_num, i % image_cols_num);
	Mat matResults;//保存测试结果
	num_knn->findNearest(sample_mat, K, matResults);//knn分类预测    
													//cout << matResults << endl;
	return (int)matResults.at<float>(0, 0);
}

void getTreeTop(cv::Mat &_img, cv::Point &center, bool &state)      //俯视看树桩
{

	/*  土黄色  */

	int iLowH = 7;
	int iHighH = 30;

	int iLowS = 17;
	int iHighS = 122;

	int iLowV = 106;
	int iHighV = 255;


	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  

	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = false;

	int minDist = _img.rows*_img.rows + _img.cols*_img.cols;   //选择最上面的红色部分
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 1.2, minRatio = 0.8, ratio = double(width) / height;

		//cout << height << ',' << width << ',' << ratio << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height > 250 || width > 250)  continue;
		if (height < 50 || width < 50)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;

		//cout << height << ',' << width << ',' << ratio << endl;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		cv::Point tempCenter = cv::Point(px + width / 2, py + height / 2);
		int tempDist = sqrt(pow((tempCenter.x - _img.cols / 2), 2) + pow((tempCenter.y - _img.rows / 2), 2));

		state = true;

		if (minDist > tempDist)
		{
			minDist = tempDist;
			center = tempCenter;
		}
	}

	if (state == true)
	{
		//cv::Mat showImg = _img.clone();
		//circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
		//cv::imshow("showImg", showImg);
		//cv::waitKey(1);
	}

}


map<int, Point2f> get_down_result(Mat src_image, Ptr<KNearest> num_knn)
{
	//Mat binary_image = down_image_init(src_image).clone();

	Mat hsv_image = down_image_hsv_init(src_image).clone();
	map<int, image_params> num_roi_almost = image_split(hsv_image);
	// knn识别  
	map<int, image_params>::iterator iter;
	map<int, Point2f> num_rect;

	//cout << num_roi_almost.size() << endl;
	//static int image_name = 0;
	for (iter = num_roi_almost.begin(); iter != num_roi_almost.end(); iter++)
	{
		imshow("test", iter->second.roi);
		waitKey();
		Mat roi = down_num_roi_process(iter->second.roi);
		//cout << 1 << endl;
		if (roi.empty())
		{

			//cout << 1 << endl;
			//num_rect.insert(pair<int, Point2f>(10, iter->second.center));
		}
		else
		{
			//cout << 2 << endl;
			int result = down_num_predict(roi, num_knn);
			cout << "down:" << result << endl;
			num_rect.insert(pair<int, Point2f>(result, iter->second.center));
			//imwrite("dataset/" + to_string(image_name) + ".jpg", roi);
			//image_name++;


		}

	}
	return num_rect;
}