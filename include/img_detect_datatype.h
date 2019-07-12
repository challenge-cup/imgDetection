#pragma once 

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

enum DetectState {  //图像检测状态
	NODATA = -1, 
	HASNOT = 0,  
	HAS = 1,  
};

enum WallLocation {		//墙的左右
	RIGHT_WALL = 0,
	LEFT_WALL = 1,
};

enum NumDirection {		//标志牌方向
	STANDING = 0,
	LYING = 1,
    SIMPLE = 2,   //只检测白版
};

enum ThresholdMode {    //二值化方法
    GRAYSCALE = 0,
    HSV = 1,
};

struct image_params_fd {
	cv::Point2f center;
	cv::Mat roi;
};

int const black_thresh_fd = 20;
int const white_thresh_fd = 180;
int const white_thresh_paizi_fd = 170;

int const image_cols_fd = 20;
int const image_rows_fd = 30;

int const num_lowest_fd = 50;
float const width_height_ratio_fd = 1.5;
int const minArea_fd = 200;

// 外部接口使用参数
struct ParamGetNum {
	int numDirect;
	int wantedNum; 
	cv::Point center;
	float depth;
	bool state;

	ParamGetNum(int _numDirect, int _wantedNum, cv::Point _center, float _depth, bool _state) :
		numDirect(_numDirect),
		wantedNum(_wantedNum),
		center(_center),
		depth(_depth),
		state(_state)
		{ }
};

struct ParamGeneralDetection {
    bool state;
    cv::Point center;

    ParamGeneralDetection(bool _state, cv::Point _center) :
        center(_center),
        state(_state)
    { }
};

typedef ParamGeneralDetection ParamGenDetect;

//检测连续性 记录最远一个牌子的中心以及第几个
struct ParamCircleFarthestDetection {
    bool newFalg;
    cv::Point center;

    ParamCircleFarthestDetection(bool _newFalg, cv::Point _center) :
        center(_center),
        newFalg(_newFalg)
    { }
};

typedef ParamGeneralDetection ParamGenDetect;

// 内部检测参数
struct ParamHSV {
    cv::Scalar scalarL;
    cv::Scalar scalarH;

    ParamHSV()  {};
    ParamHSV(cv::Scalar _scalarL, cv::Scalar _scalarH):
        scalarL(_scalarL),
        scalarH(_scalarH)
    { }
};

struct ParamBIN {
    int binThreshold;

    ParamBIN()  {};
    ParamBIN(int binThreshold):
        binThreshold(binThreshold)
    { }
};

struct ParamMorph {
    bool openOp;    //是否执行开闭区间操作
    bool closeOp;    //是否执行开闭区间操作
    bool openFirst;
    int opSize, clSzie;
    int shape;

    ParamMorph() {};
    ParamMorph(int _size, bool _openFirst = true, bool _openOp = true, bool _closeOp = true, int _shape = cv::MorphShapes::MORPH_RECT) :
        opSize(_size),
        clSzie(_size),
        openOp(_openOp),
        closeOp(_closeOp),
        openFirst(_openFirst),
        shape(_shape)
    { }

    ParamMorph(int _opSize, int _clSize, bool _openFirst = true, bool _openOp = true, bool _closeOp = true, int _shape = cv::MorphShapes::MORPH_RECT) :
        opSize(_opSize),
        clSzie(_clSize),
        openOp(_openOp),
        closeOp(_closeOp),
        openFirst(_openFirst),
        shape(_shape)
    { }
};

struct ParamPreprocess {
    int thresholdMode;
    bool usingMorph;
    ParamHSV pHSV;
    ParamBIN pBIN;
    ParamMorph pMorph;

    ParamPreprocess() {};

    //使用HSV方式,使用开闭区间运算
    ParamPreprocess(ParamHSV _pHSV, ParamMorph _pMorph) :
        pHSV(_pHSV),
        pMorph(_pMorph),
        thresholdMode(ThresholdMode::HSV),
        usingMorph(true)
        { }

    //使用灰度二值化方式,使用开闭区间运算
    ParamPreprocess(ParamBIN _pBIN, ParamMorph _pMorph) :
        pBIN(_pBIN),
        pMorph(_pMorph),
        thresholdMode(ThresholdMode::GRAYSCALE),
        usingMorph(true)
        { }

    //使用HSV方式,不使用开闭区间运算
    ParamPreprocess(ParamHSV _pHSV) :
        pHSV(_pHSV),
        thresholdMode(ThresholdMode::HSV),
        usingMorph(false)
        { }

    //使用灰度二值化方式,不使用开闭区间运算
    ParamPreprocess(ParamBIN _pBIN) :
        pBIN(_pBIN),
        thresholdMode(ThresholdMode::GRAYSCALE),
        usingMorph(false)
        { }

};

typedef ParamPreprocess ParamPrep;

struct ParamContourSelect {
    int  minW, maxW, minH, maxH;
    double minRatio, maxRatio;

    ParamContourSelect() {};
    ParamContourSelect(int _minW, int _maxW, int _minH, int _maxH, float _minRatio, float _maxRatio) :
        minW(_minW),
        maxW(_maxW),
        minH(_minH),
        maxH(_maxH),
        minRatio(_minRatio),
        maxRatio(_maxRatio)
    { }
};

typedef std::vector<std::vector<cv::Point>> contourType;
typedef std::vector<cv::Rect> typeVecRect;
