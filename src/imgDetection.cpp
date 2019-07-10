#include "imgDetection.h"

using namespace cv;
using namespace cv::ml;

void imgDetection::fullScreenNum(cv::Mat &_img, cv::Mat &dst, std::vector<cv::Point> &vertex)
{
	//Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[])  ;
	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(_img.cols - 1, 0);
	corners[2] = Point2f(_img.cols - 1, _img.rows - 1);
	corners[3] = Point2f(0, _img.rows - 1);
	vector<Point2f> corners_trans(4);
	for (int i = 0; i <= 3; i++)    {
		corners_trans[i] = vertex[i];
		//cout << "circleCenter:" << i << "=" << "(" << corners_trans[i].x << "," << corners_trans[i].y << ")" << endl;
	}
	Mat M = getPerspectiveTransform(corners_trans, corners);

	warpPerspective(_img, dst, M, Size(0, 0));

	//imshow("3", dst);
	//waitKey(0);	

}

void imgDetection::trainModel()
{
	Mat data, labels; //data和labels分别存放  
	Mat tepImage;

	// 读取0-9的样本图片
	for (int i = 0; i <= 10; i++)   {
		std::string imgPath = "../data/bottom/" + std::to_string(i) + ".jpg";		//训练集图片位置
		tepImage = imread(imgPath, IMREAD_GRAYSCALE);					//灰度图形式读取，后面才能转换 CV_32F格式，无参数读取，转换有问题
		threshold(tepImage, tepImage, 180, 200.0, THRESH_BINARY);
		data.push_back(tepImage.reshape(0, 1));         				//将图像转成一维数组插入到data矩阵中
		labels.push_back(i);             								//将图像对应的标注插入到labels矩阵中
	}
	data.convertTo(data, CV_32F);										//转换 CV_32F格式，此方法的库要求必须用CV_32F格式
	cout << "hello1" << endl;
	Mat trainData, trainLabel;
	trainData = data(Range(0, 11), Range::all());
	trainLabel = labels(Range(0, 11), Range::all());

	//使用KNN算法  
	int k = 1;	//每个数字只有一个样本，所以k=1最合适，相当于就是与那个样本误差最小就选谁

	Ptr<TrainData>   tData = TrainData::create(trainData, ROW_SAMPLE, trainLabel); //ROW_SAMPLE表示一行一个样本  
	model->setDefaultK(k); model->setIsClassifier(true);
	model->train(tData);

	std::string knnPath = "..\\data\\tuxiang\\hello5.xml";
	model->save(knnPath);

	cout << "hello3" << endl;
}

void imgDetection::detectNum(Mat &_img, int &predNumber, float &distance)
{
	Mat img, grayimage;
	Mat testData;

	cv::resize(_img, img, Size(640, 480), INTER_NEAREST);
	//imshow("img", img);
	//waitKey(0);

	testData = img.reshape(0, 1);						//将图像转成一维数组插入到data矩阵中   
														//testData = img.clone();						//将图像转成一维数组插入到data矩阵中 
	cvtColor(testData, testData, COLOR_BGR2GRAY);
	testData.convertTo(testData, CV_32F);
	Mat matResults(0, 0, CV_32F);//保存测试结果 
	Mat dist;
	//float r = model1->predict(testData);
	float r = model1->findNearest(testData, 1, matResults, noArray(), dist);
	predNumber = (int)r;
	distance = dist.at<float>(0, 0);

}

void imgDetection::getNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, float &depth, bool &state)
{
    cv::Mat imgThresholded;

    ParamPrep paramPrep = ParamPrep(ParamHSV(Scalar(0, 0, 40), Scalar(180, 10, 255)), ParamMorph(7, 4));
    ParamContourSelect paramSelect = ParamContourSelect(20, 280, 50, 250, 0.8, 1.2);
    if (numDirect == NumDirection::LYING) {
        paramPrep.pHSV.scalarL = Scalar(0, 0, 40);
        paramPrep.pHSV.scalarH = Scalar(180, 10, 255);
        paramSelect = ParamContourSelect(20, 280, 20, 250, 0.88, 1.4);
    }
    else  {
        paramPrep.pHSV.scalarL = Scalar(0, 0, 184);
        paramPrep.pHSV.scalarH = Scalar(42, 3, 229);
        paramSelect = ParamContourSelect(50, 200, 20, 120, 1.2, 1.8);
    }

    imgPreprocess(_img, imgThresholded, paramPrep);
    cv::imshow("imgThresholded", imgThresholded);
    cv::waitKey(0);

    typeVecRect resRects;
    contourSelect(imgThresholded, resRects, paramSelect);

    state = false;
    std::vector<cv::Point> depthDetectArea;
    /***  simple状态，有白框直接返回  ***/
    if (numDirect == NumDirection::SIMPLE) {
        if (!resRects.empty()) {

            center = cv::Point(resRects.at(0).x + resRects.at(0).width / 2,
                                resRects.at(0).y + resRects.at(0).height / 2);
            depthDetectArea.push_back(center);
            depthDetectArea.push_back(center - cv::Point(3, 3));
            depthDetectArea.push_back(center + cv::Point(3, 3));

            getStandBoradDepth(_img, depthDetectArea, depth);

            state = true;
        }
    }
    /*   检测内部数字   */
    else {
        Mat imageROI;
        for (auto &resRect : resRects) {
            imageROI = _img(resRect); //切出数字区域
            getRoiNum(imageROI, numDirect, wantedNum, center, depthDetectArea, state);//对标志牌区域识别数字

            if (state == true) {
                // 正确时计算深度，并且getRoiNum()里用小图得到的中心值需要偏移
                for (auto &p : depthDetectArea) {
                    p.x += resRect.x;
                    p.y += resRect.y;
                }
                center.x += resRect.x;
                center.y += resRect.y;

                getStandBoradDepth(_img, depthDetectArea, depth);
                break;
            }
        }
    }

    //if (state == true) {
    //    Mat showImg = _img.clone();
    //    circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
    //    cv::imshow("showImg", showImg);
    //    cv::waitKey(1);
    //}


}

void imgDetection::imgPreprocess(const cv::Mat &_src, cv::Mat &_dst, const ParamPreprocess &_param)
{
	/*  二值化  */
    if (_param.thresholdMode == ThresholdMode::HSV) {
        cvtColor(_src, _dst, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
        cv::inRange(_dst, _param.pHSV.scalarL, _param.pHSV.scalarH, _dst); //Threshold the image 
    }
    else {
        cvtColor(_src, _dst, COLOR_RGB2GRAY);
        threshold(_dst, _dst, _param.pBIN.binThreshold, 255, THRESH_BINARY);
    }
    //cv::imshow("img", _dst);
    //cv::waitKey(0);

    /*  开闭运算  */
    if (_param.usingMorph == true) {
        Mat opElement = getStructuringElement(_param.pMorph.shape, Size(_param.pMorph.clSzie, _param.pMorph.clSzie));
        Mat clElement = getStructuringElement(_param.pMorph.shape, Size(_param.pMorph.opSize, _param.pMorph.opSize));
        if (_param.pMorph.openFirst) {
            if (_param.pMorph.openOp)
                morphologyEx(_dst, _dst, MORPH_OPEN, opElement);
            if (_param.pMorph.closeOp)
                morphologyEx(_dst, _dst, MORPH_CLOSE, clElement);
        }
        else {
            if (_param.pMorph.closeOp)
                morphologyEx(_dst, _dst, MORPH_CLOSE, clElement);
            if (_param.pMorph.openOp)
                morphologyEx(_dst, _dst, MORPH_OPEN, opElement);
        }

        //imshow("Thresholded Image2", _dst); //show the thresholded image
        //闭操作 (连接一些连通域)
    }
	//cv::imshow("_dst", _dst);
	//cv::waitKey(0);

}

void imgDetection::getRoiNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, std::vector<cv::Point> &depthDetectArea, bool &state)
{
	//Mat imageROI;
	//imageROI = _img(Rect(0, _img.rows / 2, _img.cols, _img.rows / 2)); //切出下半部分

	Mat gray_img, thresholded_img, final_img;//stand用二值化,lying用hsv

											 /*  黑色  */
    Mat imgThresholded;//stand用二值化,lying用hsv
    ParamPrep paramPrep;
    ParamContourSelect paramSelect;
    if (numDirect == NumDirection::LYING) {
        paramPrep = ParamPrep(ParamHSV(Scalar(0, 0, 0), Scalar(180, 255, 100)), ParamMorph(2));
        paramSelect = ParamContourSelect(10, 280, 10, 250, 0.4, 0.8);
    }
    else{
        paramPrep = ParamPrep(ParamBIN(150), ParamMorph(2));
        paramSelect = ParamContourSelect(7, 60, 12, 80, 0.4, 0.7);
    } 

    imgPreprocess(_img, imgThresholded, paramPrep);
    cv::Mat imgThresholdedRGB;
    cvtColor(imgThresholded, imgThresholdedRGB, COLOR_GRAY2RGB);
    cv::imshow("imgThresholded1", imgThresholded);
    cv::waitKey(0);

    typeVecRect resultRects;
    contourSelect(imgThresholded, resultRects, paramSelect);
    for (auto &rect : resultRects) {
        std::vector<cv::Point> vertex;
        vertex.push_back(cv::Point(rect.x + 1, rect.y + 1));
        vertex.push_back(cv::Point(rect.x + rect.width - 2, rect.y + 1));
        vertex.push_back(cv::Point(rect.x + rect.width - 2, rect.y + rect.height - 2));
        vertex.push_back(cv::Point(rect.x + 1, rect.y + rect.height - 2));

        Mat dst;
        fullScreenNum(imgThresholdedRGB, dst, vertex);
        //cv::imshow("dst", dst);
        //cv::waitKey(0);

        int predNum;
        float distance;
        detectNum(dst, predNum, distance); //模型位置在imgDectction.h里改
        //cout << predNum << endl;
        //cout << distance << endl;

        if (distance < 4e9)
            cout << predNum << endl;
        if (predNum == wantedNum
            && distance < 4e9)  { //KNN距离阈值
            center.x = rect.x + rect.width / 2;
            center.y = rect.y + rect.height / 2;
            state = HAS;
            //cout << predNum << "," << wantedNum << endl;
            //circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
            //cv::imshow("showImg", showImg);
            //cv::waitKey(1);

            depthDetectArea.push_back(center);
            depthDetectArea.push_back(cv::Point(rect.x, rect.y));
            depthDetectArea.push_back(cv::Point(rect.x + rect.width, rect.y + rect.height));
            //cout << depthDetectArea << endl;
        }
    }
}

void imgDetection::getStandBoradDepth(cv::Mat &_img, const std::vector<cv::Point> &P, float &boardDepth)
{

	/*  白色  */
    Mat imgThresholded;//stand用二值化,lying用hsv
    ParamPrep paramPrep = ParamPrep(ParamHSV(Scalar(0, 0, 150), Scalar(180, 10, 255)),
                                    ParamMorph(5, false, true));
    imgPreprocess(_img, imgThresholded, paramPrep);
    //cv::imshow("imgThresholded", imgThresholded);
    //cv::waitKey(0);

	int boradMin_y = imgThresholded.rows, boradMin_x = imgThresholded.cols, boradMax_y = 0, boradMax_x = 0;
	bool finishFlag = false;

	int offset = 3;   //为防止数字框外还有黑点，设置的偏移量
	//向下
	for (int i = P[2].y + offset; i < imgThresholded.rows; i++) {
		//获取第 i行首像素指针 
		uchar * p = imgThresholded.ptr<uchar>(i);
		//对第i 行的每个像素(byte)操作 
		if (p[P[0].x] == 0) {
			boradMax_y = i;
			break;
		}
	}

	//向上
	for (int i = P[1].y - offset; i >= 0; i--)  {
		//获取第 i行首像素指针 
		uchar * p = imgThresholded.ptr<uchar>(i);
		//对第i 行的每个像素(byte)操作 
		if (p[P[0].x] == 0) {
			boradMin_y = i;
			break;
		}
	}

	//向右

	for (int j = P[2].x + offset; j < imgThresholded.cols; j++) {
		for (int i = P[1].y; i <= P[2].y; i++)  {
			uchar * p = imgThresholded.ptr<uchar>(i);
			if (p[j] == 0)  {
				boradMax_x = j;
				finishFlag = true;
				break;
			}
		}
		if (finishFlag == true) break;
	}

	finishFlag = false;
	for (int j = P[1].x - offset; j >= 0; j--)  {
		for (int i = P[1].y; i <= P[2].y; i++)  {
			uchar * p = imgThresholded.ptr<uchar>(i);
			if (p[j] == 0)
			{
				boradMin_x = j;
				finishFlag = true;
				break;
			}
		}
		if (finishFlag == true) break;
	}

	//cout << boradMin_y << "," << boradMax_y << "," << boradMax_x << endl;

	//Mat showImg = _img.clone();
	//circle(showImg, cv::Point(boradMin_x,boradMin_y), 2, Scalar(255, 255, 255), 3, 8, 0);
	//circle(showImg, cv::Point(boradMax_x, boradMax_y), 2, Scalar(255, 255, 255), 3, 8, 0);
	//imshow("showImg", showImg);
	//waitKey(0);

	if (boradMax_y == 0 || boradMax_x == 0 
        || boradMin_y == imgThresholded.rows || boradMin_x == imgThresholded.cols)  {

		boardDepth = -1;
		printf("牌子在边缘，深度估计不准！已置-1\n");
		return;
	}

    cout << boradMax_y << "," << boradMin_y << "," << boradMax_x << "," << boradMin_x << endl;
    boardDepth = calcDepth(boradMax_x - boradMin_x, boradMax_y - boradMin_y);

}

float imgDetection::calcDepth(int width, int height) {
    Eigen::Matrix3f K;
    K << 269.5, 0, 319.5,
        0, 269.5, 239.5,
        0, 0, 1;

    Eigen::Vector3f X;
    X << (float)(height), (float)(width), 0.0f;

    Eigen::RowVector3f Y;
    Y = K.inverse()*X;

    float boardDepth = (0.7 / Y[0] + 1.3 / Y[1]) / 2;
    //cout << "boardDepth" << boardDepth << endl;

    return boardDepth;
}

void imgDetection::readAruco()
{
	std::ifstream infile;
	infile.open("D:\\aruco.txt");
	if (!infile.is_open()) std::cout << "no aruco.txt!" << std::endl;

	std::string s;
	while (std::getline(infile, s))
	{
		int a = std::atoi(s.c_str());
		arucoIds.push_back(a);
	}
	infile.close();             //关闭文件输入流 
	printf("读取了%d个二维码id\n", arucoIds.size());
	//for (auto iter = arucoIds.begin(); iter != arucoIds.end(); iter++)
	//	cout << *iter << endl;
}

//柱子旁边的小树苗
void imgDetection::getSapling(cv::Mat & _img, ParamGenDetect & _param)
{
    Mat imgThresholded;

    //设置不同的开闭运算核
    ParamPrep paramPrep1 = ParamPrep(ParamHSV(Scalar(36, 0, 0), Scalar(64, 255, 255)), ParamMorph(8));
    ParamPrep paramPrep2 = ParamPrep(ParamHSV(Scalar(36, 0, 0), Scalar(64, 255, 255)), ParamMorph(3));

    cv::cvtColor(_img, imgThresholded, COLOR_BGR2HSV);
    //二值化
    cv::inRange(imgThresholded, paramPrep1.pHSV.scalarL, paramPrep1.pHSV.scalarH, imgThresholded);
    //cv::imshow("imgThresholded2", imgThresholded);
    //cv::waitKey(0);
    //开闭运算
    if (paramPrep2.usingMorph == true) {
        int _size1 = paramPrep1.pMorph.clSzie;
        int _size2 = paramPrep2.pMorph.opSize;
        Mat element1 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size1, _size1));
        Mat element2 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size2, _size2));
        //先开再闭，去除噪点
        if (paramPrep2.pMorph.openOp)
            morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element2);
        //imshow("open", imgThresholded); //show the thresholded image
        //闭操作 (连接一些连通域)，防止内部轮廓出错
        if (paramPrep2.pMorph.closeOp)
            morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element1);
        //cv::imshow("close", imgThresholded);
    }
    //找最外层次的轮廓 cv::RETR_EXTERNAL
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
    //外接矩形
    cv::Mat imageContours = cv::Mat::zeros(imgThresholded.size(), CV_32FC3); //最小外接矩形画布
    cv::Rect rect = cv::Rect(0, 0, 0, 0);

    for (int i = 0; i < contours.size(); i++)
        //for (int i = 0; i < filterRect.size(); i++)
    {
        cv::Rect tRect = cv::Rect(0, 0, 0, 0);
        //cv::Rect lRect = cv::Rect(0, 0, 0, 0);
        bool calabash = false;
        int tarea;
        int farea;
        tRect = cv::boundingRect(contours[i]);
        cv::rectangle(imageContours, tRect, Scalar(0, 0, 255), 2, 8, 0);
        tarea = tRect.area();
        if (tarea < 500)
        {
            continue;
        }
        cout << "tarea  " << tarea << endl;
        if (tRect.area() > rect.area())
        {
            rect = cv::Rect(tRect);
            _param.state = true;
            cv::drawContours(imageContours, contours, i, cv::Scalar(0, 255, 0), 1, 8, hierarchy);
        }
        else
        {
            cv::drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
        }

    }
    //画轮廓 以及外接矩形
    cv::rectangle(imageContours, rect, Scalar(0, 255, 0), 2, 8, 0);
    //cv::imshow("imageContour", imageContours);
    if (_param.state) {
        _param.center.x = rect.x + rect.width / 2;
        _param.center.y = rect.y + rect.height / 2;
    }
    //_param.center.x = rect.x + rect.width / 2;
    //_param.center.y = rect.y + rect.height / 2;

}

void imgDetection::detectAruco(cv::Mat &A, cv::Mat &depthImg, int &rightMarkerIds, std::ofstream &_outfile, bool &state, int &wantId)
{
	state = false;

	Mat B;
	Mat thA;
	cvtColor(A, B, COLOR_BGR2GRAY);
	equalizeHist(B, thA);
	vector< int > markerIds;
	vector< vector<Point2f> > markerCorners, rejectedCandidates;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
	//params->adaptiveThreshWinSizeMin = 5;
	//params->adaptiveThreshWinSizeMax = 50;
	//params->polygonalApproxAccuracyRate = 0.15;
	params->minMarkerPerimeterRate = 0.0008;

	if (wantId == 759)
	{
		////二值化
		//Mat imgHSV;
		//cvtColor(_img, imgHSV, COLOR_BGR2HSV);

		cv::Mat imgThresholded, imgThresholded1, imgThresholded2;

		//灰度二值化滤波
		//cv::imshow("imgThresholded1", B);
		//cv::threshold(B, imgThresholded2,200,255, THRESH_BINARY);
		//cv::threshold(B, imgThresholded2, 200, 255, THRESH_BINARY_INV);
		//cv::imshow("imgThresholded2", imgThresholded2);

		//HSV滤波
		ParamPrep paramPrep1 = ParamPrep(ParamHSV(Scalar(0, 0, 199), Scalar(35, 6, 255)), ParamMorph(2));
		ParamPrep paramPrep2 = ParamPrep(ParamHSV(Scalar(0, 0, 199), Scalar(35, 6, 255)), ParamMorph(5));
		cvtColor(A, imgThresholded1, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
		cv::inRange(imgThresholded1, paramPrep2.pHSV.scalarL, paramPrep2.pHSV.scalarH, imgThresholded2); //Threshold the image 
		//cv::imshow("imgThresholded1", imgThresholded1);
		//cv::waitKey(0);

		if (paramPrep2.usingMorph == true) {
			int _size1 = paramPrep1.pMorph.opSize;
			int _size2 = paramPrep2.pMorph.clSzie;
			Mat element1 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size1, _size1));
			Mat element2 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size2, _size2));
			//zz
			if (paramPrep2.pMorph.openOp)
				morphologyEx(imgThresholded2, imgThresholded2, MORPH_OPEN, element1);
			if (paramPrep2.pMorph.closeOp)
				morphologyEx(imgThresholded2, imgThresholded2, MORPH_CLOSE, element2);

		}
		cv::imshow("imgThresholded2", imgThresholded2);
		//cv::waitKey(0);
		//imgThresholded = imgThresholded1 | imgThresholded2;
		imgThresholded = imgThresholded2;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
		//外接矩形
		cv::Rect fRect = cv::Rect(0, 0, 0, 0);
		cv::Rect rect = cv::Rect(0, 0, 0, 0);
		cv::Mat imageContours = cv::Mat::zeros(imgThresholded.size(), CV_32FC3); //最小外接矩形画布
		for (int i = 0; i < contours.size(); i++)
		{
			cv::Rect tRect = cv::Rect(0, 0, 0, 0);
			//cv::Rect lRect = cv::Rect(0, 0, 0, 0);
			int tarea;
			int farea;
			tRect = cv::boundingRect(contours[i]);
			tarea = tRect.area();
			if (tarea < 2000)
			{
				continue;
			}
			rect = fRect | tRect;
			fRect = cv::Rect(rect);
			cv::rectangle(imageContours, rect, Scalar(0, 0, 255), 2, 8, 0);

		}
		cv::rectangle(imageContours, rect, Scalar(0, 255, 0), 2, 8, 0);
		cv::rectangle(A, rect, Scalar(0, 255, 0), 2, 8, 0);
		cv::imshow("contours", imageContours);

		cv::Mat secondImg = imgThresholded1(rect);
		ParamPrep paramPrep3 = ParamPrep(ParamHSV(Scalar(0, 0, 0), Scalar(61, 255, 141)), ParamMorph(2));
		cv::inRange(secondImg, paramPrep3.pHSV.scalarL, paramPrep3.pHSV.scalarH, secondImg); //Threshold the image 
		//cv::imshow("secondImg", secondImg);
		//cv::imshow("secondImg2", secondImg);
		std::vector<std::vector<cv::Point>> contours1;
		std::vector<cv::Vec4i> hierarchy1;
		findContours(secondImg, contours1, hierarchy1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
		//外接矩形
		//cv::Rect wantRect;
		cv::Rect sRect = cv::Rect(0, 0, 0, 0);
		cv::Rect sfRect = cv::Rect(0, 0, 0, 0);
		cv::Mat imageContours1 = cv::Mat::zeros(secondImg.size(), CV_32FC3); //最小外接矩形画布
		cout <<" contours1.size()"<< contours1.size() << endl;
		for (int i = 0; i < contours1.size(); i++)
			//for (int i = 0; i < filterRect.size(); i++)
		{
			
			//cv::Rect lRect = cv::Rect(0, 0, 0, 0);
			cv::Rect stRect = cv::Rect(0, 0, 0, 0);
			//cv::Rect lRect = cv::Rect(0, 0, 0, 0);
			int sfarea;
			stRect = cv::boundingRect(contours1[i]);
			if (stRect.area() < 2000)
			{
				continue;
			}
			sRect = sfRect | stRect;
			sfRect = cv::Rect(sRect);
			cv::rectangle(imageContours1, sRect, Scalar(0, 0, 255), 2, 8, 0);
			cv::imshow("imageContours1", imageContours1);
		}
		cout << "zzzzz" << sRect.x+rect.x << "	" << sRect.y+ rect.y << endl;
		cv::rectangle(secondImg, sRect, 200, 2, 8, 0);
		cv::imshow("secondImg", secondImg);
		//给坐标
		std::vector<cv::Point2f> wantMarkerCorners;
		wantMarkerCorners.push_back(cv::Point2f(sRect.x + rect.x, sRect.y + rect.y));
		wantMarkerCorners.push_back(cv::Point2f(sRect.x + rect.x + sRect.height, sRect.y + rect.y));
		wantMarkerCorners.push_back(cv::Point2f(sRect.x + rect.x + sRect.height, sRect.y + rect.y + sRect.width));
		wantMarkerCorners.push_back(cv::Point2f(sRect.x + rect.x, sRect.y + rect.y + sRect.width));
		markerCorners.push_back(wantMarkerCorners);
		markerIds.push_back(759);
	}
	else
	{
		cv::aruco::detectMarkers(thA, dictionary, markerCorners, markerIds, params, rejectedCandidates);
	}
	



	if (markerIds.size() != 0) {
		rightMarkerIds = markerIds[0];
		printf("有二维码:%d\n", rightMarkerIds);
	}
	else	{
		printf("没有任何二维码\n");
		return;
	}
	vector<cv::Point2f> rightmarkerCorners = markerCorners[0];
	int maxMarkerSize = 0;

	//用最大对角距离筛选多个二维码
	if (markerCorners.size() > 0)   {
		for (int i = 0; i <= markerCorners.size() - 1; i++) {
			int markerSize = abs(markerCorners[i][2].x - markerCorners[i][0].x) + abs(markerCorners[i][2].y - markerCorners[i][0].y);  //曼哈顿距离
			if (markerSize > maxMarkerSize)
			{
				maxMarkerSize = markerSize;
				rightMarkerIds = markerIds[i];
				rightmarkerCorners = markerCorners[i];
			}
		}
		//std::cout << rightMarkerIds << std::endl;
		//std::cout << rightmarkerCorners << std::endl;
		//std::cout << markerIds.size() << std::endl;
		aruco::drawDetectedMarkers(A, markerCorners);
	}

	cv::Point rightMarkerCenter = cv::Point((int)(rightmarkerCorners[2].x + rightmarkerCorners[0].x) / 2,
		(int)(rightmarkerCorners[2].y + rightmarkerCorners[0].y) / 2);

	if (depthImg.at<float>(rightMarkerCenter) >= 254.0f)    {
		printf("二维码不在有深度的柱子上\n");
		state = false;
		return;
	}

	//float * p = imgThresholded.ptr<float>(i);
	////对第i 行的每个像素(byte)操作 
	//for (int j = 0; j < imgThresholded.cols; ++j)
	//	if (p[j]<200)grayA.at<float>(i, j) = 255;
	for (auto iter = alreadyMarkerIds.begin(); iter != alreadyMarkerIds.end(); iter++)  {
		if (rightMarkerIds == *iter)    {
			printf("已经记录过了，溜了");
			state = true;
			return;
		}
	}
	//检验二维码的值是否对应要求的值
	int i = 0;
	for (auto iter = arucoIds.begin(); iter != arucoIds.end(); iter++)  {
		i++;
		if (rightMarkerIds == *iter)    {
			state = true;
			//result.txt 

			static int count = 0;
			if (count != 0) _outfile.open("D:\\result.txt", std::ios::app);
			count++;

			_outfile << std::to_string(rightMarkerIds) << ' '
				<< std::to_string((int)(rightmarkerCorners[0].x)) << ' '
				<< std::to_string((int)(rightmarkerCorners[0].y)) << ' '
				<< std::to_string((int)rightmarkerCorners[2].x) << ' '
				<< std::to_string((int)rightmarkerCorners[2].y) << ' '
				<< std::endl;
			_outfile.close();

			//rightMarkerIds.png
			imwrite("D:\\images\\" + std::to_string(rightMarkerIds) + ".png", A);
			//rightMarkerIds.data
		}
	}

	//FindAruco();             //开始记录.pfm

	//imshow("HSV", imgThresholded);
	//imshow("A",A);
	//imshow("gray", grayA);
}

void imgDetection::getCircleDown(cv::Mat &_img, cv::Point &center, bool &state, cv::Rect &CircleBottom)
{
    /***     HSV     ***/
    Mat imgThresholded, imgThresholded1, imgThresholded2;

    ParamPrep paramPrep1 = ParamPrep(ParamHSV(Scalar(156, 10, 194), Scalar(180, 255, 255)), ParamMorph(5, false));
    ParamPrep paramPrep2 = ParamPrep(ParamHSV(Scalar(0, 10, 194), Scalar(13, 255, 255)), ParamMorph(5, false));

    imgPreprocess(_img, imgThresholded1, paramPrep1);
    imgPreprocess(_img, imgThresholded2, paramPrep2);
    imgThresholded = imgThresholded1 | imgThresholded2;//  红色分布在两个hsv区域内，分别筛选后合起来
    //cv::imshow("imgThresholded", imgThresholded);
    //cv::waitKey(0);

    /***     轮廓筛选     ***/
    typeVecRect resRects;
    ParamContourSelect paramSelect = ParamContourSelect(30, 640, 5, 180, 1.7, 20);
    contourSelect(imgThresholded, resRects, paramSelect);

    state = false;
    int minY = imgThresholded.rows - 1;   //选择最上面的红色部分
    for (auto &resRect : resRects) {
        cv::Point tempCenter = cv::Point(resRect.x + resRect.width / 2, resRect.y + resRect.height / 2);
        state = true;
        //cout << predNum << "," << wantedNum << endl;
        if (minY > tempCenter.y) {
            minY = tempCenter.y;
            center = tempCenter;
        }
    }
    cv::Rect fRect = cv::Rect(0, 0, 0, 0);
    cv::Rect rect = cv::Rect(0, 0, 0, 0);
    for (int i = 0; i < resRects.size(); i++)
        //for (int i = 0; i < filterRect.size(); i++)
    {
        cv::Rect tRect = cv::Rect(resRects[i]);
        //cv::Rect lRect = cv::Rect(0, 0, 0, 0);
        bool calabash = false;
        int tarea;
        int farea;
        tarea = tRect.area();
        //if (tarea < 20)
        //{
        //    continue;
        //}
        rect = fRect | tRect;
        fRect = cv::Rect(rect);
    }
    CircleBottom = cv::Rect(rect);
    //cout << rect.x << '\t' << rect.y << '\t' << rect.height << '\t' << rect.width << endl;
    cv::Mat imageContours = cv::Mat::zeros(imgThresholded.size(), CV_32FC3); //最小外接矩形画布
    cv::rectangle(imageContours, rect, Scalar(0, 255, 0), 2, 8, 0);
    cv::rectangle(imgThresholded, CircleBottom, 150, 2, 8, 0);
    cv::imshow("imgThresholded", imgThresholded);
    cv::imshow("imageContours", imageContours);
    cv::waitKey(0);
    //if (state == true)
    //{
    //	cv::Mat showImg = _img.clone();
    //	circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
    //	cv::imshow("showImg", showImg);
    //	cv::waitKey(0);
    //}
}
//void imgDetection::getTreeTop(cv::Mat &_img, ParamGenDetect &_param) {   //俯视看树桩
//    getTreeTop(_img, _param.center, _param.state);
//}      

void imgDetection::getTreeTop(cv::Mat &_img, cv::Point &center, bool &state)      //俯视看树桩
{
    cv::Mat imgThresholded;

    /*  土黄色  */
    ParamPrep paramPrep = ParamPrep(ParamHSV(Scalar(7, 17, 106), Scalar(30, 122, 255)), ParamMorph(5, false));
    imgPreprocess(_img, imgThresholded, paramPrep);

    typeVecRect resRects;
    ParamContourSelect paramSelect = ParamContourSelect(50, 250, 50, 250, 0.8, 1.2);
    contourSelect(imgThresholded, resRects, paramSelect);

    //选择距离中心最近的树桩
    state = false;
    int minDist = _img.rows*_img.rows + _img.cols*_img.cols;   
    int minY = imgThresholded.rows - 1;   
    for (auto &resRect : resRects) {
        cv::Point tempCenter = cv::Point(resRect.x + resRect.width / 2, resRect.y + resRect.height / 2);
        int tempDist = sqrt(pow((tempCenter.x - _img.cols / 2), 2) + pow((tempCenter.y - _img.rows / 2), 2));

        state = true;
        if (minDist > tempDist) {
            minDist = tempDist;
            center = tempCenter;
        }
    }

	//if (state == true)  {
	//	cv::Mat showImg = _img.clone();
    //  cout << "center : " << center << endl;
	//	circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
	//	cv::imshow("showImg", showImg);
	//	cv::waitKey(0);
	//}

}

//连续性 zz
void imgDetection::getBottomDown(cv::Mat & _img, ParamCircleFarthestDetection &_param, std::vector<cv::Rect> &allCircle, std::vector<int> &directFlag)
{
    ////二值化
    //Mat imgHSV;
    //cvtColor(_img, imgHSV, COLOR_BGR2HSV);
    Mat imgThresholded, imgThresholded1, imgThresholded2;
    //红色
    ParamPrep paramPrep1 = ParamPrep(ParamHSV(Scalar(0, 44, 196), Scalar(15, 255, 255)), ParamMorph(2));
    ParamPrep paramPrep2 = ParamPrep(ParamHSV(Scalar(0, 44, 196), Scalar(15, 255, 255)), ParamMorph(4));

    //imgPreprocess(_img, imgThresholded1, paramPr4ep1);
    //imgPreprocess(_img, imgThresholded2, paramPrep2);
    //(const cv::Mat &_src, cv::Mat &_dst, const ParamPreprocess &_param)

    cvtColor(_img, imgThresholded2, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
    cv::inRange(imgThresholded2, paramPrep2.pHSV.scalarL, paramPrep2.pHSV.scalarH, imgThresholded2); //Threshold the image 

    //cv::imshow("imgThresholded2", imgThresholded2);
    //cv::waitKey(0);

    /*  开闭运算  */
    if (paramPrep2.usingMorph == true) {
        int _size1 = paramPrep1.pMorph.opSize;
        int _size2 = paramPrep2.pMorph.clSzie;
        Mat element1 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size1, _size1));
        Mat element2 = getStructuringElement(paramPrep2.pMorph.shape, Size(_size2, _size2));
        //zz

        if (paramPrep2.pMorph.openOp)
            morphologyEx(imgThresholded2, imgThresholded2, MORPH_OPEN, element1);
        if (paramPrep2.pMorph.closeOp)
            morphologyEx(imgThresholded2, imgThresholded2, MORPH_CLOSE, element2);
        //cv::imshow("close", imgThresholded2);
        if (paramPrep2.pMorph.openOp)
            morphologyEx(imgThresholded2, imgThresholded2, MORPH_OPEN, element1);
        //imshow("Thresholded Image2", _dst); //show the thresholded image
        //闭操作 (连接一些连通域)
    }
    //cv::imshow("_dst", imgThresholded2);
    //cv::waitKey(0);
    //imgThresholded = imgThresholded1 | imgThresholded2;
    imgThresholded = imgThresholded2;

    ////找牌子轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
    //外接矩形
    cv::Mat imageContours = cv::Mat::zeros(imgThresholded.size(), CV_32FC3); //最小外接矩形画布
    //int numFlag = 0;
    //std::vector<cv::Rect> recordRect;
    cv::Rect tempRect;
    //int sameThreshold = 20;
    std::vector<cv::Rect> filterRect;
    cout << "contours.size" << contours.size() << endl;
    cv::Rect fRect = cv::Rect(0, 0, 0, 0);
    cv::Rect rect = cv::Rect(0, 0, 0, 0);
    bool newCircleFlag = false;
    if (contours.size() == 0)
    {
        _param.newFalg = false;
    }

    for (int i = 0; i < contours.size(); i++)
        //for (int i = 0; i < filterRect.size(); i++)
    {
        cv::Rect tRect = cv::Rect(0, 0, 0, 0);
        //cv::Rect lRect = cv::Rect(0, 0, 0, 0);
        bool calabash = false;
        int tarea;
        int farea;
        tRect = cv::boundingRect(contours[i]);
        tarea = tRect.area();
        if (tarea < 50)
        {
            continue;
        }
        rect = fRect | tRect;
        fRect = cv::Rect(rect);
        //一直没有牌子，再出现牌子，则认为新的牌子出现
        if (_param.newFalg == false)
        {
            _param.newFalg = true;
            newCircleFlag = true;
        }

    }

    int width, height, area;
    width = rect.width;
    area = rect.area();
    //cout << "矩形坐标" << " x: " << rect.x << " y: " << rect.y << " 面积是：" << area << endl;
    if (allCircle.size() != 0)
    {
        tempRect = cv::Rect(allCircle.back());
        cout << tempRect.x << "\t" << tempRect.y << "\t" << tempRect.width << "\t" << tempRect.height << endl;

        cout << "newCircleFlag:" << newCircleFlag << endl;
        if (newCircleFlag)
        {
            cout << "allCircle.back:	" << allCircle.back().x << "\t" << allCircle.back().y << endl;
            _param.center.x = allCircle.back().x;
            _param.center.y = allCircle.back().y;
            //比较前后两个牌子左右关系
            if (tempRect.x > rect.x)
            {
                directFlag.push_back(-1);
            }
            else
            {
                directFlag.push_back(1);
            }
            allCircle.push_back(rect);
        }

    }
    else
    {
        allCircle.push_back(rect);
        //newCircleFlag = false;
    }
    //cout << "筛选之后矩形坐标" << "Circle number " << _param.number << "x:" << rect.x << "y:" << rect.y << "面积是：" << area << endl << endl;	
    //cout << height << ',' << width << ',' << ratio << endl;
    //drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
    //cv::imshow("imageContours", imageContours);
    //cv::waitKey(0);
    //cout << height << ',' << width << ',' << ratio << endl;
    //drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
    cv::rectangle(imageContours, rect, Scalar(0, 255, 0), 2, 8, 0);
    //cv::line(imageContours, Point(0, 120), Point(639, 120), Scalar(0, 0, 255), 2, 8, 0);
    //cv::line(imageContours, Point(0, 180), Point(639, 180), Scalar(0, 0, 255), 2, 8, 0);
    //cv::imshow("contours", imageContours);

    //directFlag.push_back(true);
    //for (auto val : directFlag) {
    //	cout << "directFlag:" << val << '\t';
    //}
    //_param.number = numFlag;

    //allCircle[0] = recordRect[0];



}

void imgDetection::contourSelect(const cv::Mat &_imgThresholded, typeVecRect &_rect, const ParamContourSelect &_param) {  // 矩形形状筛选

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(_imgThresholded, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point());
    std::vector<cv::RotatedRect> box(contours.size());

    cv::Mat imageContours = cv::Mat::zeros(_imgThresholded.size(), CV_32FC3); //最小外接矩形画布 

    for (int i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        int width, height;
        width = rect.width;
        height = rect.height;

        float ratio = double(width) / height;
        //cout << height << ',' << width << ',' << ratio << endl;
        //drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
        //cv::imshow("imageContours", imageContours);
        //cv::waitKey(0);

        //筛选形状
        if (height > _param.maxH || width > _param.maxW)  continue;
        if (height < _param.minH || width < _param.minW)  continue;
        if ((ratio < _param.minRatio) || (ratio > _param.maxRatio))	 continue;

        cout << height << ',' << width << ',' << ratio << endl;
        drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
        cv::imshow("img1", imageContours);
        cv::waitKey(0);
        _rect.push_back(rect);
    }
}  