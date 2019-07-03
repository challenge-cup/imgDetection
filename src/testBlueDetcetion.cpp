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

	for (int i = 1; i <= 12; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/blue/" + to_string(i) + ".jpg";
		img = imread(path);	//待处理图片放在这

		bool state;
		WallLocation wallLocation = WallLocation::LEFT_WALL;

		imgDetector.blueDetection(img, state, wallLocation);      //得到是否有想要的数字，需输入牌子的方向
		if (state == true)
		{
			cout << "find blue part!" << endl;
			if (wallLocation == WallLocation::LEFT_WALL)		cout << "wallLocation:left " << endl;
			else if (wallLocation == WallLocation::RIGHT_WALL)	cout << "wallLocation:right " << endl;


		}
		//cv::imshow("img", img);
		//cv::waitKey(0);
	}

	system("pause");
	return 0;
}
