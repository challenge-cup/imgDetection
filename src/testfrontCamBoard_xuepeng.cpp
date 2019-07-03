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

	for (int i = 1; i <= 8; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/lying/" + to_string(i) + ".jpg";
		img = imread(path);	//待处理图片放在这

		int wantedNum = 6;
		cv::Point center = cv::Point(0,0);
		bool state = false;

		imgDetector.get_fd_util(img, wantedNum, center, state);

		if (state == true)	cout << "find num " << wantedNum << ", center: " << center << endl;

		//cv::imshow("img", imageROI);
		//cv::waitKey(0);
	}

	system("pause");

	return 0;
}
