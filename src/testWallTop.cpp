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

int main()	{
	cv::Mat img, fullScreenNumImg;
	imgDetection imgDetector;

	for (int i = 2; i <= 3; i++)	{
		cout << "processing img " << i << endl;
		std::string path = "../data/wallTop/" + to_string(i) + ".jpg";
		img = imread(path);									//待处理图片放在这

		bool state = false;
		int centerX = 0;

		imgDetector.getWallTop(img, state, centerX);      //得到是否有想要的数字，需输入牌子的方向
		if (state == true)		{
			cout << "find wallTop part!" << " centerX:" << centerX << endl;
		}
		//cv::imshow("img", img);
		//cv::waitKey(0);
	}
	system("pause");
	return 0;
}
