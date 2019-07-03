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

	for (int i = 1; i <= 30; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/stand/" + to_string(i) + ".jpg";
		img = imread(path);	//待处理图片放在这

		ParamGetNum param(NumDirection::STANDING, 6, cv::Point(0, 0), 0, false);
		imgDetector.getNum(img, param);

		cout << "state: " << param.state << " depth: " << param.depth << endl<< endl;

		cv::imshow("img", img);
		cv::waitKey(0);
	}

	system("pause");

	return 0;
}
