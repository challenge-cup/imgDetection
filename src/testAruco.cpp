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

	for (int i = 2; i <= 2; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/QRcode/" + to_string(i) + ".jpg";
		img = imread(path);	//待处理图片放在这
		int rightMarkerIds = 0;
		ofstream outfile("result.txt", ios::app);
		bool state = false;

		//imgDetector.findrect(cv::Mat _img, int numDirect);
		imgDetector.detectAruco(img, rightMarkerIds, outfile, state);

		cout << state << endl<< endl;

		cv::imshow("img", img);
		cv::waitKey(0);
	}

	system("pause");

	return 0;
}
