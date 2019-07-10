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

int roi();

void distCorrect(cv::Mat &src, cv::Mat &dst)
{
	imshow("src", src);
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 533.6104383276903;
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(0, 2) = 314.7377259417282;
	cameraMatrix.at<double>(1, 1) = 530.391280459703;
	cameraMatrix.at<double>(1, 2) = 263.2418306925699;
	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = -0.419344758615447;
	distCoeffs.at<double>(1, 0) = 0.1928393891655499;
	distCoeffs.at<double>(2, 0) = -0.003557686801411734;
	distCoeffs.at<double>(3, 0) = -0.0007128083296022629;
	distCoeffs.at<double>(4, 0) = 0;
	Mat view, rview, map1, map2;
	Size imageSize;
	imageSize = src.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	remap(src, dst, map1, map2, INTER_LINEAR);

	
	//imshow("dst", dst);
	//waitKey(0);

}

int main()
{

	int a = roi();

	//cv::Mat img, fullScreenNumImg;
	//imgDetection imgDetector;

	//for (int i = 1; i <= 8; i++)
	//{
	//	cout << "processing img " << i << endl;
	//	std::string path = "../data/lying/" + to_string(i) + ".jpg";
	//	img = imread(path);	//待处理图片放在这

	//	int wantedNum = 6;
	//	cv::Point center = cv::Point(0,0);
	//	bool state = false;

	//	imgDetector.get_fd_util(img, wantedNum, center, state);

	//	if (state == true)	cout << "find num " << wantedNum << ", center: " << center << endl;

	//	//cv::imshow("img", imageROI);
	//	//cv::waitKey(0);
	//}

	system("pause");

	return 0;
}

int roi()
{
	//std::string file_name = "D:/qrcode/1 (4).png";
	//std::string file_name = "C:/Users/goodluck/Desktop/barrier/0 (92).png";
	//Mat src = imread(file_name, 1);
	//std::string out_dir = "D:/qrcode/1 (4).png";
	//std::string out_dir= "C:/Users/goodluck/Desktop/prosamples/0 (92).png";
	//int iLowH = 0;//24-102
	//int iHighH = 42;

	string file_name = "D:\imgDetection\data\quad\1.png";


	//int iLowS = 0;//0-108
	//int iHighS = 11;

	int iLowH = 0;
	int iHighH = 15;

	int iLowS = 44;
	int iHighS = 255;

	int iLowV = 75;
	int iHighV = 255;

	int index = 1;
	int index_max = 50;
	namedWindow("Control", cv::WINDOW_FREERATIO); //create a window called "Control"  
											  //Create trackbars in "Control" window  
	cv::createTrackbar("LowH", "Control", &iLowH, 180); //Hue (0 - 179)  
	cv::createTrackbar("HighH", "Control", &iHighH, 180);

	cv::createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)  
	cv::createTrackbar("HighS", "Control", &iHighS, 255);
	cv::createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)  
	cv::createTrackbar("HighV", "Control", &iHighV, 255);
	cv::createTrackbar("index", "Control", &index, index_max);
	Mat imgHSV;
	vector<Mat> hsvSplit;
	double con_area = 0;
	do {
		//file_name = "D:/qrcode/1 (" + std::to_string(index + 1) + ").png";
		//file_name = "../data/quad/"+ std::to_string(index + 1) + ".png";93
        file_name = "../data/Aruco759/" + std::to_string(index + 359) + ".png";
        //std::string path = "../data/stand/" + std::to_string(index + 1) + ".jpg";
		try {
			auto img = imread(file_name);
			//distCorrect(img, img);
			cvtColor(img, imgHSV, COLOR_BGR2HSV);
			//split(imgHSV, hsvSplit);
			//equalizeHist(hsvSplit[2], hsvSplit[2]);
			//merge(hsvSplit, imgHSV);

			//cout << "iLowH: " << iLowH << "iHighH:" << iHighH << "iLowS: " << iLowS << "iHighS:" << iHighS << "iLowV: " << iLowV << "iHighV:" << iHighV << endl;
			Mat imgThresholded;
			inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(1, 1));
			//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
			morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

			/*ecust::CVUtils::removeSmallRegion(imgThresholded, imgThresholded, 100, 1, 1);*/
			//vector<vector<Point>> contours;
			//vector<Vec4i> hierarchy;
			//findContours(imgThresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
			//vector<RotatedRect> box(contours.size());

			//double area = 0;
			//double max_area = 0;
			//Mat imageContours = Mat::zeros(img.size(), CV_32FC3);
			//Mat dest;
			//for (int i = 0; i < contours.size(); i++)
			//{
			//	Rect rect = cv::boundingRect(contours[i]);
			//	int width, height, px, py;
			//	px = rect.x;
			//	py = rect.y;
			//	width = rect.width;
			//	height = rect.height;
			//	area = width * height;

			//	//if (height > 200 || width > 200)continue;
			//	if (height < 12 || width < 12)continue;
			//	if (double(height) / width > 2 || double(width) / height > 2)continue;
			//	if (max_area > area)continue;
			//	max_area = area;
			//	cv::Mat dst;
			//	dst = img(cv::Rect(px, py, width, height));
			//	dest = dst;
			//	//imshow(std::to_string(i), dst); //show the original image
			//	//imwrite(out_dir + std::to_string(index) + "_" + to_string(i) + ".png", dst);
			//}

			imshow("Thresholded Image", imgThresholded); //show the thresholded image  
														 //imshow("dest", dest);											 //imshow("imageContours", imageContours); //show the original image  
			imshow("Original", img); //show the original image  
									 //std::cout << index << endl;
			//cv::waitKey(0);
			char key = (char)waitKey(300);

			//if (key == 27)
			//break;
		}
		catch (cv::Exception e)
		{

		}
		// index = index + 1;
	} while (index<=index_max);
	return 0;
}