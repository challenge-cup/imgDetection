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

	cv::Mat img, fullScreenNumImg;
	imgDetection imgDetector;
	cv::Rect CircleBottom;//返回下视圈的矩形

	for (int i = 1; i <= 310; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/circleDown/1 (" + to_string(i) + ").png";
		img = imread(path);	//待处理图片放在这

        ParamGenDetect _param(false, cv::Point(0, 0));
        imgDetector.getCircleDown(img, _param, CircleBottom);      //得到是否有想要的数字，需输入牌子的方向

		//打印两个center 看区别
		cout << CircleBottom.x << '\t' << CircleBottom.y << '\t' << CircleBottom.height << '\t' << CircleBottom.width << endl;
		cout << "center1 : " << CircleBottom.x + CircleBottom.width / 2 << '\t' << CircleBottom.y + CircleBottom.height / 2 << endl;

        cout << "state : " << _param.state << endl;
        if (_param.state == true)   cout << "center : " << _param.center << endl;

		cv::imshow("img", img);
		cv::waitKey(0);
	}

	system("pause");

	return 0;
}
