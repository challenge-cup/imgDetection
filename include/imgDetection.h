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

#include "img_detect_datatype.h"

using namespace std;
using namespace Eigen;

class imgDetection {

public:

    /**  新检测接口  **/
    // 检测数字
	inline void getNum(cv::Mat &_img, ParamGetNum &param)  {
        getNum(_img, param.numDirect, param.wantedNum, param.center, param.depth, param.state);
    }
    //检测树桩顶部
    inline void getTreeTop(cv::Mat &_img, ParamGenDetect &_param) {
        getTreeTop(_img, _param.center, _param.state);
    }
    //检测圈顶部
    inline void getCircleDown(cv::Mat &_img, ParamGenDetect &_param, cv::Rect &CircleBottom) {
        getCircleDown(_img, _param.center, _param.state, CircleBottom);
    }

    void getBottomDown(cv::Mat & _img, ParamCircleFarthestDetection &_param, std::vector<cv::Rect> &allCircle, std::vector<int> &directFlag);
    //getBottomDown(_img, _param.center, _param.number);

    /*******旧版检测接口*******/

    //得到是否有想要的数字，需输入牌子的方向
    void getNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, float & depth, bool &state);
    void getCircleDown(cv::Mat &_img, cv::Point &center, bool &state, cv::Rect &CircleBottom);				//俯视看圈
    void getTreeTop(cv::Mat &_img, cv::Point &center, bool &state);					//俯视看树桩


	void trainModel();		//数字检测模型训练

	//Aruco
    void imgDetection::getSapling(cv::Mat & _img, ParamGenDetect & _param);
	void readAruco();
	void detectAruco(cv::Mat &A, cv::Mat &depthImg, int &rightMarkerIds, std::ofstream &_outfile, bool &state,int &wantId);
    std::vector<int> alreadyMarkerIds;
    std::vector<int> arucoIds;

private:
	cv::Point2f  rectVertex[4];
	cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();               //待训练模型
	cv::Ptr<cv::ml::KNearest> model1 = cv::ml::StatModel::load<cv::ml::KNearest>("./hello5.xml");
	cv::Ptr<cv::ml::KNearest> num_knn = cv::ml::StatModel::load<cv::ml::KNearest>("./down_knn_param.xml"); //前置模型
    
    /******* 内部图像处理函数 *******/

    void imgPreprocess(const cv::Mat &_src, cv::Mat &_dst, const ParamPreprocess &_param);  // HSV+开闭区间
    void contourSelect(const cv::Mat &_imgThresholded, typeVecRect &_rect,const ParamContourSelect &_param);  // 矩形形状筛选
    void getStandBoradDepth(cv::Mat &_img, const std::vector<cv::Point> &P, float &boardDepth);
    float calcDepth(int width, int height);
    void getRoiNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center,
                    std::vector<cv::Point> &depthDetectArea, bool &state);  //小块图像检测数字

	void detectNum(cv::Mat &_img, int &predNumber, float &distance);	//数字检测
	void fullScreenNum(cv::Mat &_img, cv::Mat &dst, std::vector<cv::Point> &vertex);						//数字全屏放大

};

#endif 

