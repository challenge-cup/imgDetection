#include <iostream>  
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/ml/ml.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include <Eigen/Core>
#include <Eigen/Dense>

#include "imgDetection.h"

using namespace std;
using namespace cv;
using namespace chrono;

int main()
{

	cv::Mat img, tempImage;
	imgDetection imgDetector;
	std::vector<cv::Rect> allCircle;//存储十个圆圈的坐标
	std::vector<int> directFlag;//存储寻找第n个圈的左右方向 false左边 true右边
	ParamCircleFarthestDetection _param(false, cv::Point(0, 0));	

	for (int i = 105; i <= 142; i++)
	{
		cout << endl << endl << "processing img " << i << endl;
		std::string path = "../data/bottomDown3/" + to_string(i) + ".png";
		img = imread(path);	//待处理图片放在这		

		Rect rect(0, 100, img.size().width, 70);//改了以下切的区域
		Mat image_roi = img(rect);
        imgDetector.getBottomDown(image_roi, _param, allCircle, directFlag);      //根据每帧图像返回坐标

        cout << "state : " << _param.newFalg << endl;
        //if (_param.newFalg != 0)   cout << "center : " << _param.center << endl;

		for (auto val : directFlag) {
			cout << "directflag:" << val << '\t';
		}


		cv::imshow("img", img);
		cv::imshow("image_roi", image_roi);
		cv::waitKey(0);
	}


	//测试例子
	//cout << endl << endl << "processing img " << 170 << endl;
	//std::string path = "../data/bottomDown/" + to_string(170) + ".png";
	//cv::Mat	img;
	//img= imread(path);
	//cv::Rect r1 = cv::Rect(0, 0, 50, 80);
	//cv::Rect r2 = cv::Rect(100, 100, 20, 20);
	//r2 = r1 | r2;
	//cv::Rect r3 = cv::Rect(r1);

	///*r1.height = 4;*/
	////cout << r3.area()<< r3.x << r3.y << r3.height << r3.width;
	//cv::Mat imageContours = cv::Mat::zeros(img.size(), CV_32FC3);
	//cv::rectangle(imageContours, r1, Scalar(255, 0, 0), 2, 8, 0);
	//cv::rectangle(imageContours, r2, Scalar(0, 255, 0), 2, 8, 0);
	//cv::rectangle(imageContours, r3, Scalar(0, 0, 255), 2, 8, 0);
	//cv::imshow("contours", imageContours);
	//cv::waitKey(0);
	//r3 = r3;

	system("pause");

	return 0;
}
