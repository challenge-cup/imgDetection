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

#include "windows.h"

using namespace std;
using namespace cv;
using namespace chrono;

int main()
{

	cv::Mat img, fullScreenNumImg;
	imgDetection imgDetector;
	//cv::Rect CircleBottom;//返回下视圈的矩形

	for (int i = 315; i <= 384; i++)
	{
		cout << "processing img " << i << endl;
		//std::string path = "../data/circleDown/1 (" + to_string(i) + ").png";
		std::string path = "../data/sapling1/" + std::to_string(i) + ".png";
		img = imread(path);	//待处理图片放在这

        ParamGenDetect _param(false, cv::Point(0, 0));
        imgDetector.getSapling(img, _param);      //是否看到树，看到的话返回中心坐标

		//打印两个center 看区别
		//cout << CircleBottom.x << '\t' << CircleBottom.y << '\t' << CircleBottom.height << '\t' << CircleBottom.width << endl;
		//cout << "center1 : " << CircleBottom.x + CircleBottom.width / 2 << '\t' << CircleBottom.y + CircleBottom.height / 2 << endl;
		if (_param.state) {
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN);
		}
		else
		{
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
		}
        cout << "state : " << _param.state << endl;
        if (_param.state == true)   cout << "center : " << _param.center << endl;

		cv::imshow("img", img);
		cv::waitKey(0);
	}

	system("pause");

	return 0;
}
