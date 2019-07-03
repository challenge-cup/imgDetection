//#pragma once 
#ifndef IMG_DETECTION_H
#define IMG_DETECTION_H

//#include <stdafx.h>  
#include <iostream>  
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>  
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc/imgproc.hpp>  

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef PI
#define PI 3.1415926535
#endif

using namespace std;
using namespace Eigen;

enum DetectState {
	NODATA = -1, //无图片
	HASNOT = 0,  //未检测到
	HAS = 1,  //检测到
};

enum NumDirection {
	STANDING = 0,  //未检测到
	LYING = 1,  //检测到
};

struct image_params_fd {
	cv::Point2f center;
	cv::Mat roi;
};

class imgDetection {

public:
	cv::Point2f  rectVertex[4];
	cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();

	// 检测状态

	void rectPreprocess(cv::Mat &src, cv::Mat &dst);			//矩形检测预处理
	void detectRectangle(cv::Mat &_img);					//矩形检测
	void detectRectangleP(cv::Mat &_img);					//矩形检测（概率霍夫）
	void detectRectangleC(cv::Mat &_img);					//分割
	void removeSmallRegion(cv::Mat& src, cv::Mat& dst, int areaLimit, int checkMode, int neighborMode);

	void getrectVertex(std::vector<cv::Vec2f> &_lines, cv::Mat &_img);	//矩形交点计算
	void fullScreenNum(cv::Mat &_img);								//数字全屏放大
	void fullScreenNum(cv::Mat &_img, cv::Mat &dst, std::vector<cv::Point> &vertex);						//数字全屏放大

	void getNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, float & depth, bool &state);      //得到是否有想要的数字，需输入牌子的方向
	void getLyingBorad(cv::Mat &_img, cv::Point &center, bool &state);
	void getStandBoradDepth(cv::Mat &_img, std::vector<cv::Point> P, float &boardDepth);

	void trainModel();										//数字检测模型训练
	void detectNum(cv::Mat &_img, int &predNumber, float &distance);	//数字检测

	cv::Point detectCircleRGB(cv::Mat &_img, int &state);
	cv::Point detectCircle(cv::Mat &_img, int &radius, int &state);       //圆检测
	void getCircleParam(cv::Mat &_img, cv::Point &circleCenter, int &rightAvgDepth, int &leftAvgDepth, float &xyLengthRatio, bool &state);

	cv::Point detectOval(cv::Mat &_img, int &state);
	void Computer_axy(std::vector<cv::Point> contour, cv::Size imgSize, int &state);  //计算椭圆的半长轴，中心点
	int hough_ellipse(std::vector<cv::Point> contour);				//霍夫变换计算椭圆的theta,b
	cv::Mat draw_Eliipse(cv::Mat);									//画椭圆

	void readAruco();
	void detectAruco(cv::Mat &A, cv::Mat &depthImg, int &rightMarkerIds, std::ofstream &_outfile, bool &state);
	std::vector<int> alreadyMarkerIds;
	std::vector<int> arucoIds;
	
	void get_fd_util(cv::Mat &_img, int wantedNum, cv::Point &center, bool &state);
	void getCircleDown(cv::Mat &_img, cv::Point &center, bool &state);      //俯视看圈
	void blueDetection(cv::Mat &_img, bool &state);      //检查蓝色边缘是否出现

private:
	double oval_a;    //半长轴
	double oval_b;    //短轴
	double oval_theta;   //椭圆的旋转角度
	cv::Point oval_center;   //椭圆中心的坐标   
	cv::Ptr<cv::ml::KNearest> model1 = cv::ml::StatModel::load<cv::ml::KNearest>("./hello5.xml");
	cv::Ptr<cv::ml::KNearest> num_knn = cv::ml::StatModel::load<cv::ml::KNearest>("./down_knn_param.xml"); //前置模型
	
																										   //front_down
	int const black_thresh_fd = 20;
	int const white_thresh_fd = 180;
	int const white_thresh_paizi_fd = 170;

	int const image_cols_fd = 20;
	int const image_rows_fd = 30;

	int const num_lowest_fd = 50;
	float const width_height_ratio_fd = 1.5;
	int const minArea_fd = 200;

	map<int, cv::Point2f> get_fd_result(cv::Mat src_img);
	cv::Mat image_init_fd(cv::Mat src_img);
	vector<cv::Rect> findrect_fd(cv::Mat half_binary_img);
	map<int, image_params_fd> get_num_roi_almost_fd(vector<cv::Rect> feature_rect, cv::Mat src_img);
	cv::Mat num_roi_process_fd(cv::Mat num_roi_almost);
	int num_predict_fd(cv::Mat num_roi, cv::Ptr<cv::ml::KNearest> num_knn);

	void getRoiNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, std::vector<cv::Point> &depthDetectArea, bool &state);
	vector<cv::Rect> findrect(cv::Mat &_img, int numDirect);
	void getHSVwhite(cv::Mat &src, cv::Mat &dst);
};

#endif 

