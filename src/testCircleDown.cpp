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

//int main()
//{
//
//	cv::Mat img, fullScreenNumImg;
//	imgDetection imgDetector;
//	cv::Rect CircleBottom;//返回下视圈的矩形
//
//	for (int i = 1577; i <= 1696; i++)
//	{
//		cout << "processing img " << i << endl;
//		//std::string path = "../data/circleDown/1 (" + to_string(i) + ").png";
//		//std::string path = "../data/circleDownBig/" + std::to_string(i) + ".png";
//		std::string path = "../data/circleDownBig/" + std::to_string(i) + ".png";
//		img = imread(path);	//待处理图片放在这
//
//        ParamGenDetect _param(false, cv::Point(0, 0));
        //imgDetector.getCircleDown(img, _param, CircleBottom);      //得到是否有想要的数字，需输入牌子的方向
//
//		//打印两个center 看区别
//		cout << CircleBottom.x << '\t' << CircleBottom.y << '\t' << CircleBottom.height << '\t' << CircleBottom.width << endl;
//		cout << "center1 : " << CircleBottom.x + CircleBottom.width / 2 << '\t' << CircleBottom.y + CircleBottom.height / 2 << endl;
//
//        cout << "state : " << _param.state << endl;
//        if (_param.state == true)   cout << "center : " << _param.center << endl;
//
//		cv::imshow("img", img);
//		cv::waitKey(0);
//	}
//
//	system("pause");
//
//	return 0;
//}
double getDistance(cv::Point2d point1, cv::Point2d point2);
void swapVector(std::vector <cv::Point2d>&v, int a, int b);
double getDistanceStraight(cv::Point2d point1, cv::Point2d point2);
int getTime(cv::Point2d point1, cv::Point2d point2);

std::vector <cv::Point2d> points = std::vector <cv::Point2d>{ cv::Point2d(0, 0),cv::Point2d(0,-12.5),cv::Point2d(0,50),cv::Point2d(1,30),cv::Point2d(0,-3.5),cv::Point2d(0,15) };
int main()
{
	cv::Point2d;
	//cv::Point2d treePoint2d = cv::Point2d(0, 0);
	//std::vector <cv::Point2d> Point2ds = std::vector <cv::Point2d>{ cv::Point2d(0, 0),cv::Point2d(13,4),cv::Point2d(18,5),cv::Point2d(1,9),cv::Point2d(18,3),cv::Point2d(5,7) };
	//std::vector <cv::Point2d> points = std::vector <cv::Point2d>{ cv::Point2d(0, 0),cv::Point2d(0,-12.5),cv::Point2d(0,9.8),cv::Point2d(0,9.7),cv::Point2d(0,-3.5),cv::Point2d(0,15) };

	for (int i = 0; i < 6; i++)
	{
		std::cout << points[i] << "		";
	}
	std::cout << points.size() - 1 << " " << std::endl;
	double shortestDistance = 0;
	double tempDistance = 0;
	int tempTime = 0;
	int shortestTime = 0;
	//shortestDistance = getDistance(points[0], points[1]);
	for (int i = 0; i < points.size() - 1; i++)
	{
		//std::cout << points[i] ;
		for (int j = i + 1; j < points.size(); j++)
		{
			tempDistance = getDistanceStraight(points[i], points[j]);
			tempTime = getTime(points[i], points[j]);
			if (j == i + 1)
			{
				shortestDistance = tempDistance;
				shortestTime = tempTime;
			}
			if (tempTime < shortestTime)
			{
				shortestDistance = tempDistance;
				shortestTime = tempTime;
				swapVector(points, i + 1, j);
			}
			else if (tempTime == shortestTime)
			{

				if (tempDistance < shortestDistance)
				{
					shortestDistance = tempDistance;
					swapVector(points, i + 1, j);
				}
			}
/*			if (tempDistance < shortestDistance)
			{
				shortestDistance = tempDistance;
			}*/			
		}
	}
	for (int i = 0; i < 6; i++)
	{
		std::cout << points[i] << "		";
	}
	system("pause");
	return 0;
}
double getDistance(cv::Point2d point1, cv::Point2d point2) {
	double distance;
	distance = powf((point1.x - point2.x), 2) + powf((point1.y - point2.y), 2);
	distance = sqrtf(distance);
	return distance;
}
void swapVector(std::vector <cv::Point2d>&v, int a, int b) {
	cv::Point2d temp = v[a];
	v[a] = v[b];
	v[b] = temp;
}

double getDistanceStraight(cv::Point2d point1, cv::Point2d point2)
{
	double distance;
	distance = abs(point1.x - point2.x) + abs(point1.y - point2.y);
	return distance;
}
int getTime(cv::Point2d point1, cv::Point2d point2){
	int timeX,timeY;
	//>10 7s <10 5s
	timeX = abs(point1.x - point2.x) > 10 ? 7 : (abs(point1.x - point2.x) == 0 ? 0 : 5);
	timeY = abs(point1.y - point2.y) > 10 ? 7 : (abs(point1.y - point2.y) == 0 ? 0 : 5);
	return timeX + timeY;
}

