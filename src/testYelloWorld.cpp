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
#include "new_down.h"
#include "new_forward.h"

using namespace std;
using namespace cv;
using namespace chrono;


int main()
{

	cv::Mat img, correctImg, correctDownImg;
	Ptr<KNearest> down_num_knn = StatModel::load<KNearest>("down_knn_param.xml");
	Ptr<KNearest> forward_num_knn = StatModel::load<KNearest>("num_knn_param.xml");

	for (int i = 8; i <= 9; i++)
	{
		cout << "processing img " << i << endl;
		std::string path = "../data/dji_down/" + to_string(i) + ".png";
		//std::string path = "../data/dji_stand/1 (" + to_string(i) + ").png";
		//std::string path = "../data/dataset_forward/1 (" + to_string(i) + ").png";
		img = imread(path);	//待处理图片放在这

		resize(img, img, Size(640, 480));
		//cv::imshow("img", img);
		//cv::waitKey(0);
		/*******down*******/
		distCorrect(img, correctDownImg);
		map<int, Point2f> down_result = get_down_result(correctDownImg, down_num_knn);
		if (down_result.size() > 0) {
			for (auto down_iter = down_result.begin(); down_iter != down_result.end(); down_iter++)
			{
				cout << "find num: " << down_iter->first << endl;

				cv::Mat showImg = correctDownImg.clone();
				circle(showImg, down_iter->second, 8, Scalar(0, 0, 255), -1, 8, 0);
				cv::imshow("showImg", showImg);
				cv::waitKey(0);

			}
		}
	}

	system("pause");

	return 0;
}

