#include "imgDetection.h"

using namespace cv;
using namespace cv::ml;



Point imgDetection::detectCircleRGB(Mat &_img, int &state)
{
	if (!_img.data)
	{
		cout << "img has no data! " << endl;
		state = NODATA;

		Point pt(0, 0);
		return pt;
	}

	int iLowH = 156;
	int iHighH = 180;

	int iLowS = 43;
	int iHighS = 255;

	int iLowV = 46;
	int iHighV = 255;

	Mat imgHSV;
	vector<Mat> hsvSplit;
	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
										   //imshow("imgHSV", imgHSV);

	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);
	Mat imgThresholded1, imgThresholded2, imgThresholded;

	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded1);

	iLowH = 0;
	iHighH = 10;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded2);

	imgThresholded = imgThresholded1 + imgThresholded2;

	Mat image;
	cvtColor(_img, image, COLOR_BGR2GRAY);
	//高斯滤波
	GaussianBlur(image, image, Size(7, 7), 2, 2);
	Canny(imgThresholded, image, 200, 100);
	//霍夫圆
	vector<Vec3f> circles;
	HoughCircles(image, circles, HOUGH_GRADIENT, 1, 10, 200, 100, 0, 0);

	/*  一个圆有可能识别出几个相邻的层叠的圆，圆心暂时取平均处理，
	如果识别出圆太多可能需要取前几个平均提高速度，暂未考虑实际有多个圆在图像里*/

	Point center(0, 0);
	if (circles.size() == 0)
		state = HASNOT;
	else
	{
		state = HAS;

		for (size_t i = 0; i < circles.size(); i++)
			center += Point(cvRound(circles[i][0]), cvRound(circles[i][1]));

		center.x = center.x / circles.size();
		center.y = center.y / circles.size();

		circle(_img, center, 4, Scalar(255, 255, 255), -1, 8, 0);
		circle(_img, center, circles[0][2], Scalar(255, 255, 255), 3, 8, 0);

	}

	imshow("_img", _img);
	waitKey(0);

	return center;

}

Point imgDetection::detectCircle(Mat &_img, int &radius, int &state)
{

	if (!_img.data)
	{
		cout << "img has no data! " << endl;
		state = NODATA;

		Point pt(0, 0);
		return pt;
	}

	Mat image;
	/*Canny(_img, image, 200, 100);

	//融合相近轮廓
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(image, image, MORPH_CLOSE, element);

	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Mat temptImg(_img.size(), CV_8UC1, Scalar(0));

	vector<vector<cv::Point>> closeContours;

	vector<vector<cv::Point>> contours1;
	//只找圆环中封闭的小圆，外部大圆和标牌以及柱子连接不便于检测，利用该曲线不封闭剔除
	for (int i = 0; i <= contours.size()-1; i++)
	{
	if (abs(contours[i][0].x - contours[i][contours[i].size() - 1].x) <= 3
	&& abs(contours[i][0].y - contours[i][contours[i].size() - 1].y) <= 3) //封闭条件：首尾相近
	{
	Mat tempImg(_img.size(), CV_8UC1, Scalar(0));
	//closeContours.push_back(contours[i]);
	contours1.push_back(contours[i]);
	drawContours(tempImg, contours1, -1, Scalar(196));//填充反向选区
	cout << contours[i] << endl << endl;
	imshow("tempImg", tempImg);
	waitKey(0);
	}
	}

	drawContours(temptImg, closeContours, -1, Scalar(196));//填充反向选区
	imshow("temptImg", temptImg);
	waitKey(0);*/

	//霍夫圆
	vector<Vec3f> circles;
	//HoughCircles(temptImg, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 25, 0, 0);
	HoughCircles(_img, circles, HOUGH_GRADIENT, 1, 10, 200, 25, 0, 0);

	//cvtColor(_img, _img, CV_GRAY2RGB);

	Point center(0, 0);

	if (circles.size() == 0)
		state = HASNOT;
	else
	{
		state = HAS;

		for (size_t i = 0; i < circles.size(); i++)
		{
			center += Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
			radius += circles[i][2];
			//circle(_img, Point(cvRound(circles[i][0]), cvRound(circles[i][1])), circles[i][2], Scalar(0, 0, 255), 1, 2, 0);
		}
		center.x = center.x / circles.size();
		center.y = center.y / circles.size();
		radius = radius / circles.size();

	}

	//imshow("_img", _img);
	//waitKey(0);

	return center;

}

void imgDetection::getCircleParam(Mat & _img, Point &circleCenter, int & rightAvgDepth, int & leftAvgDepth, float & xyLengthRatio, bool &state)
{
	uint leftPosition = 0, rightPosition = 0, upPosition = 0, downPosition = 0;

	//过圆心水平线遍历
	uchar* pData1 = _img.ptr<uchar>(circleCenter.y);
	//计算左侧平均深度及左侧中心位置
	int meetCircleFlag = 0;
	int count = 0;
	for (int j = circleCenter.x; j >= 0; j--)
	{
		if (pData1[j] < 250) //边缘处有约251-254的数值，不收录进用来算深度均值的数据里
		{
			leftAvgDepth += pData1[j];
			if (meetCircleFlag == 0) leftPosition = j;
			count++;
			meetCircleFlag = 1;
		}
		else if (meetCircleFlag == 1)
			break; //遍历遇到圆后，下一次看到255停止遍历,防止外部还有东西
	}
	if (count != 0) leftAvgDepth /= count;

	//计算右侧平均深度及右侧中心位置
	meetCircleFlag = 0;
	count = 0;
	for (int j = circleCenter.x; j < _img.cols; j++)
	{
		if (pData1[j] < 250) //边缘处有约251-254的数值，不收录进用来算深度均值的数据里
		{
			rightAvgDepth += pData1[j];
			if (meetCircleFlag == 0) rightPosition = j;
			count++;
			meetCircleFlag = 1;
		}
		else if (meetCircleFlag == 1)
			break; //遍历遇到圆后，下一次看到255停止遍历,防止外部还有东西
	}
	if (count != 0) rightAvgDepth /= count;

	//过圆心竖直线遍历
	//计算下方平均深度及下方中心位置
	for (int i = circleCenter.y; i<_img.rows; i++)
	{
		uchar* pData1 = _img.ptr<uchar>(i);
		if (pData1[circleCenter.x] < 250) //边缘处有约251-254的数值，不收录进用来算深度均值的数据里
		{
			downPosition = i;
			break;
		}
	}

	//计算上方平均深度及上方中心位置
	for (int i = circleCenter.y; i >= 0; i--)
	{
		uchar* pData1 = _img.ptr<uchar>(i);
		if (pData1[circleCenter.x] < 250) //边缘处有约251-254的数值，不收录进用来算深度均值的数据里
		{
			upPosition = i;
			break;
		}
	}

	int xLength = 0, yLength = 0;

	//圆的上下左右缺角时处理
	state = 1;
	if (rightPosition == 0 && leftPosition == 0) //双边缺无法处理
		state = 0;
	else if (rightPosition == 0)
		xLength = 2 * (leftPosition - circleCenter.x);//单边缺用另一边两倍长
	else if (leftPosition == 0)
		xLength = 2 * (rightPosition - circleCenter.x);
	else
		xLength = rightPosition - leftPosition;

	if (upPosition == 0 && downPosition == 0)
		state = 0;
	else if (upPosition == 0)
		yLength = 2 * (downPosition - circleCenter.y);
	else if (downPosition == 0)
		yLength = 2 * (upPosition - circleCenter.y);
	else
		yLength = upPosition - downPosition;

	xyLengthRatio = abs((float)xLength / (float)yLength);


}

void imgDetection::rectPreprocess(Mat &src, Mat &dst)
{
	if (!src.data)
		cerr << "Read picture error!" << endl;

	Mat grayImg, gaussianImg;
	gaussianImg.create(src.size(), CV_8UC1);

	GaussianBlur(src, gaussianImg, Size(5, 5), 0, 0);//高斯滤波
	Canny(src, dst, 100, 50);//Canny边缘检测
	imshow("dst", dst);
	waitKey(0);

	/*imshow("resilt",dst);
	waitKey(0);*/
}

void imgDetection::getrectVertex(std::vector<Vec2f> &_lines, Mat &_img)
{
	//画出结果  
	Mat result;
	_img.copyTo(result);

	std::vector<Vec2f>::const_iterator it = _lines.begin();
	float vTheta[2], vRho[2];
	float hTheta[2], hRho[2];
	int horCount = 0, verCount = 0;

	float stdRho = (*it)[0];
	float stdTheta = (*it)[1];

	while (it != _lines.end())
	{
		float rho = (*it)[0];
		float theta = (*it)[1];
		if (abs(theta - stdTheta)< PI / 4. || abs(theta - stdTheta) > 3. *PI / 4.)
		{
			hRho[horCount] = rho;
			hTheta[horCount] = theta;
			horCount++;
		}
		else
		{
			vRho[verCount] = rho;
			vTheta[verCount] = theta;
			verCount++;
		}
		if (theta < PI / 4. || theta > 3. *PI / 4.)
		{
			Point pt1(rho / cos(theta), 0);
			Point pt2((rho - result.rows*sin(theta)) / cos(theta), result.rows);
			line(result, pt1, pt2, Scalar(255), 1);
		}
		else
		{
			Point pt1(0, rho / sin(theta));
			Point pt2(result.cols, (rho - result.cols*cos(theta)) / sin(theta));
			line(result, pt1, pt2, Scalar(255), 1);
		}
		++it;

		imshow("resilt", result);
		waitKey(0);
		cout << "(" << rho << "," << theta << ")1" << endl;

		if (verCount + horCount >= 4)
			break;

	}

	cout << "(" << hRho[0] << "," << hTheta[0] << ")" << endl;
	cout << "(" << hRho[1] << "," << hTheta[1] << ")" << endl;
	cout << "(" << vRho[0] << "," << vTheta[0] << ")" << endl;
	cout << "(" << vRho[1] << "," << vTheta[1] << ")" << endl;

	vector<Point> Pt;
	for (int i = 0; i <= 1; i++)
		for (int j = 0; j <= 1; j++)
		{
			Matrix2d rho_s, rho_c, s_c;
			rho_s << hRho[i], vRho[j],
				sin(hTheta[i]), sin(vTheta[j]);

			rho_c << cos(hTheta[i]), cos(vTheta[j]),
				hRho[i], vRho[j];

			s_c << cos(hTheta[i]), cos(vTheta[j]),
				sin(hTheta[i]), sin(vTheta[j]);

			int index = 2 * i + j;
			rectVertex[index] = Point2f(rho_s.determinant() / s_c.determinant(), rho_c.determinant() / s_c.determinant());
			circle(result, rectVertex[index], 3, Scalar(0, 255, 0), -1, 8, 0);
			/*cout << rho_s << endl << endl;
			cout << rho_c << endl << endl;
			cout << s_c << endl << endl << endl;*/
		}
	//展示结果  

	for (int i = 0; i <= 3; i++)
		cout << rectVertex[i] << "rectVertex" << endl;

	Point corner[4] = { Point(0,0),
		Point(0,_img.rows - 1),
		Point(_img.cols - 1,0),
		Point(_img.cols - 1,_img.rows - 1) };

	for (int i = 0; i <= 3; i++)
	{
		int min = INT_MAX, minloc = 0;

		for (int j = i; j <= 3; j++)
		{
			int squareDistance = pow(rectVertex[j].x - corner[i].x, 2) + pow(rectVertex[j].y - corner[i].y, 2);

			if (squareDistance < min)
			{
				min = squareDistance;
				minloc = j;
			}
			cout << squareDistance << "squareDistance" << endl;
			cout << min << "min" << endl;
		}

		swap(rectVertex[i], rectVertex[minloc]);
	}

	imshow("2", result);

	waitKey(0);


}

void imgDetection::detectRectangle(Mat &_img)
{

	std::vector<Vec2f> lines;
	//调用函数
	HoughLines(_img, lines, 1, PI / 90, 65);

	//展示结果的图像
	std::cout << "共检测出线：" << lines.size() << "条" << std::endl;
	if (lines.size() != 4)
		std::cout << "共检测出线 != 4条!" << std::endl;

	getrectVertex(lines, _img);
}

void imgDetection::detectRectangleP(Mat &_img)
{

	Mat result;
	_img.copyTo(result);

	std::vector<Vec4i> xylines;
	HoughLinesP(_img, xylines, 1, CV_PI / 180, 20, 10, 100);

	//展示结果的图像

	std::cout << "共检测出线：" << xylines.size() << "条" << std::endl;
	if (xylines.size() != 4)
		std::cout << "共检测出线 != 4条!" << std::endl;


	std::vector<Vec2f> lines;

	for (size_t i = 0; i <= xylines.size() - 1; i++)
	{
		line(result, Point(xylines[i][0], xylines[i][1]),
			Point(xylines[i][2], xylines[i][3]), Scalar(255, 255, 255), 10, 8);
		//circle(result, Point(xylines[i][0], xylines[i][1]), 10, Scalar(0,0,0), 2);
		//circle(result, Point(xylines[i][2], xylines[i][3]), 10, Scalar(0,0,0), 2);


		cout << Point(xylines[i][0], xylines[i][1]) << Point(xylines[i][2], xylines[i][3]) << endl;

		float tmpTheta = -atan((float)(xylines[i][2] - xylines[i][0]) / (float)(xylines[i][3] - xylines[i][1]));
		float tmpRho = xylines[i][0] * cos(tmpTheta) + xylines[i][1] * sin(tmpTheta);

		lines.push_back(Vec2f(tmpRho, tmpTheta));
		cout << tmpTheta << "  ,  " << tmpRho << endl;
		imshow("test", result);
		waitKey(0);
	}

	getrectVertex(lines, _img);

}


void imgDetection::detectRectangleC(Mat &_img)
{

	const int iLowH = 0;
	const int iHighH = 180;

	const int iLowS = 0;
	const int iHighS = 40;

	const int iLowV = 200;
	const int iHighV = 255;
	//std::vector<Feature> result;
	cv::Mat imgHSV;
	std::vector<cv::Mat> hsvSplit;
	auto img = _img;
	cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
	cv::split(imgHSV, hsvSplit);
	cv::equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);

	cv::Mat imgThresholded;
	inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	morphologyEx(imgThresholded, imgThresholded, cv::MORPH_OPEN, element);

	cv::imshow("imgThresholded", imgThresholded);
	cv::waitKey(0);

	////去除孔洞
	//removeSmallRegion(imgThresholded, imgThresholded, 100, 2, 1);
	////画轮廓

	//cv::imshow("imgThresholded1", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(img.size(), CV_32FC3); //最小外接矩形画布 

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		cout << width << ',' << height << endl;

		if (height > 400 || width > 400)continue;
		if (height < 12 || width < 12)continue;
		if (double(height) / width > 1.25 || double(width) / height > 1.25)continue;
		drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		box[i] = minAreaRect(contours[i]);
		std::vector<cv::Point2f> perspective_points(4);
		cv::Point2f rects[4];
		box[i].points(rects);
		std::vector<cv::Point2f> points(rects, rects + 4);
		perspective_points[2].x = rect.width;
		perspective_points[2].y = 0;
		perspective_points[3].x = rect.width;
		perspective_points[3].y = rect.height;
		perspective_points[0].x = 0;
		perspective_points[0].y = rect.height;
		perspective_points[1].x = 0;
		perspective_points[1].y = 0;
		cv::Mat transform;
		cv::Size size(rect.width, rect.height);
		transform = cv::getPerspectiveTransform(points, perspective_points);
		cv::Mat dst;
		cv::warpPerspective(img, dst, transform, size);

		cv::imshow("dst", dst);
		cv::waitKey(0);
	}

}

void imgDetection::removeSmallRegion(cv::Mat& src, cv::Mat& dst, int areaLimit, int checkMode, int neighborMode)
{
	int removeCount = 0;
	cv::Mat pointLabel = cv::Mat::zeros(src.size(), CV_8UC1);
	if (checkMode == 1)
	{
		// cout<<"Mode: 去除小区域. ";
		for (int i = 0; i < src.rows; ++i)
		{
			uchar* iData = src.ptr<uchar>(i);
			uchar* iLabel = pointLabel.ptr<uchar>(i);
			for (int j = 0; j < src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		// cout<<"Mode: 去除孔洞. ";
		for (int i = 0; i < src.rows; ++i)
		{
			uchar* iData = src.ptr<uchar>(i);
			uchar* iLabel = pointLabel.ptr<uchar>(i);
			for (int j = 0; j < src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	std::vector<cv::Point2i> neighborPos;  //记录邻域点位置
	neighborPos.push_back(cv::Point2i(-1, 0));
	neighborPos.push_back(cv::Point2i(1, 0));
	neighborPos.push_back(cv::Point2i(0, -1));
	neighborPos.push_back(cv::Point2i(0, 1));
	if (neighborMode == 1)
	{
		// cout<<"Neighbor mode: 8邻域."<<endl;
		neighborPos.push_back(cv::Point2i(-1, -1));
		neighborPos.push_back(cv::Point2i(-1, 1));
		neighborPos.push_back(cv::Point2i(1, -1));
		neighborPos.push_back(cv::Point2i(1, 1));
	}
	// else cout<<"Neighbor mode: 4邻域."<<endl;
	int NeihborCount = 4 + 4 * neighborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测
	for (int i = 0; i < src.rows; ++i)
	{
		uchar* iLabel = pointLabel.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********
				std::vector<cv::Point2i> GrowBuffer;                                      //堆栈，用于存储生长点
				GrowBuffer.push_back(cv::Point2i(j, i));
				pointLabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //检查四个邻域点
					{
						CurrX = GrowBuffer.at(z).x + neighborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + neighborPos.at(q).y;
						if (CurrX >= 0 && CurrX<src.cols&&CurrY >= 0 && CurrY<src.rows)  //防止越界
						{
							if (pointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(cv::Point2i(CurrX, CurrY));  //邻域点加入buffer
								pointLabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查
							}
						}
					}

				}
				if (GrowBuffer.size()>areaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出
				else { CheckResult = 1;   removeCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //更新Label记录
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					pointLabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********
			}
		}
	}

	checkMode = 255 * (1 - checkMode);
	//开始反转面积过小的区域
	for (int i = 0; i < src.rows; ++i)
	{
		uchar* iData = src.ptr<uchar>(i);
		uchar* iDstData = dst.ptr<uchar>(i);
		uchar* iLabel = pointLabel.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = checkMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}
}

void imgDetection::fullScreenNum(Mat &_img)
{
	//Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[])  ;
	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(0, _img.rows - 1);
	corners[2] = Point2f(_img.cols - 1, 0);
	corners[3] = Point2f(_img.cols - 1, _img.rows - 1);
	vector<Point2f> corners_trans(4);
	for (int i = 0; i <= 3; i++)
	{
		corners_trans[i] = rectVertex[i];
		cout << "circleCenter:" << i << "=" << "(" << corners_trans[i].x << "," << corners_trans[i].y << ")" << endl;

	}
	Mat M = getPerspectiveTransform(corners_trans, corners);
	Mat result;
	warpPerspective(_img, result, M, Size(0, 0));
	result.copyTo(_img);
	imshow("3", _img);

	waitKey(0);
}

void imgDetection::fullScreenNum(cv::Mat &_img, cv::Mat &dst, std::vector<cv::Point> &vertex)
{

	//Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[])  ;
	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(_img.cols - 1, 0);
	corners[2] = Point2f(_img.cols - 1, _img.rows - 1);
	corners[3] = Point2f(0, _img.rows - 1);
	vector<Point2f> corners_trans(4);
	for (int i = 0; i <= 3; i++)
	{
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
	for (int i = 0; i <= 10; i++)
	{
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

	//resize(_img, img, Size(500, 400));

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

Point imgDetection::detectOval(Mat &src, int &state)
{
	Point pt;
	state = HASNOT;

	if (!src.data)
	{
		cout << "Read image error" << endl;
		state = NODATA;
		pt = Point(0, 0);
		return pt;
	}

	//GaussianBlur(src, src, Size(3,3), 0, 0);       //高斯模糊（Gaussian Blur）

	//  namedWindow("Source", CV_WINDOW_AUTOSIZE);
	// imshow("Source", src);
	Mat imgHSV;
	vector<Mat> hsvSplit;
	cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
										  //imshow("imgHSV", imgHSV);

	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);
	Mat imgThresholded1, imgThresholded2, imgThresholded;

	int iLowH = 156;
	int iHighH = 180;
	int iLowS = 43;
	int iHighS = 255;
	int iLowV = 46;
	int iHighV = 255;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded1);

	iLowH = 0;
	iHighH = 10;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded2);

	imgThresholded = imgThresholded1 + imgThresholded2;


	Mat dst;
	cvtColor(src, dst, COLOR_RGB2GRAY);
	GaussianBlur(dst, dst, Size(3, 3), 0, 0);       //高斯模糊（Gaussian Blur）
	Canny(imgThresholded, dst, 100, 200, 3);                  //Canny边缘检测
															  //Canny(dst, dst, 100, 200, 3);
															  //namedWindow("Canny", CV_WINDOW_AUTOSIZE);
															  //imshow("Canny", dst);
															  //提取轮廓************************************************
	vector<vector<Point>>  contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//Hough变换************************************************
	int accumulate;   // Hough空间最大累积量
	imgDetection imgDetector;

	Mat result(src.size(), CV_8UC3, Scalar(0));
	for (int i = 0; i< contours.size(); i++)
	{
		int tmpstate = HAS;

		imgDetector.Computer_axy(contours[i], dst.size(), tmpstate);
		if (tmpstate == HASNOT)
			state = HASNOT;
		else
		{
			accumulate = imgDetector.hough_ellipse(contours[i]);
			if (accumulate >= contours[i].size()*0.25)    // 判断是否超过给定阈值，判断是否为椭圆
			{
				result = imgDetector.draw_Eliipse(src);
				state = HAS;
			}
			else
			{
				cout << "This profile is not an oval" << endl;
				state = HASNOT;
			}
		}
	}
	//namedWindow("Hough_result", CV_WINDOW_AUTOSIZE);
	imshow("Hough_result", result);

	waitKey(0);


}

void imgDetection::Computer_axy(vector<Point> contour, Size imgsize, int &state)
{
	float Ly, Lx, LL;
	double maxVal;
	Mat distance(1, contour.size(), CV_32FC1);      //每一点到轮廓的所有距离
	Mat max_distance(imgsize, CV_32FC1, Scalar(0));  //每一点到轮廓的最大距离

													 //  查找椭圆的上下左右界，小于一定范围不算椭圆，同时用上下左右界遍历i,j

	int xmax = 0, xmin = imgsize.width - 1, ymax = 0, ymin = imgsize.height - 1;
	for (int n = 0; n<contour.size(); n++)
	{
		if (ymax < contour.at(n).y)
			ymax = contour.at(n).y;
		if (xmax < contour.at(n).x)
			xmax = contour.at(n).x;
		if (ymin > contour.at(n).y)
			ymin = contour.at(n).y;
		if (xmin > contour.at(n).x)
			xmin = contour.at(n).x;
	}

	if (ymax - ymin <= 20 || xmax - xmin <= 20)
		state = HASNOT;
	else
	{
		for (int i = ymin; i <= ymax; i++)
		{
			for (int j = xmin; j <= xmax; j++)
			{
				for (int n = 0; n<contour.size(); n++)
				{
					Ly = (i - contour.at(n).y)*(i - contour.at(n).y);
					Lx = (j - contour.at(n).x)*(j - contour.at(n).x);
					LL = sqrt(Ly + Lx);
					distance.at<float>(n) = LL;
				}
				minMaxLoc(distance, NULL, &maxVal, NULL, NULL);
				max_distance.at<float>(i, j) = maxVal;
			}
		}

		Mat cut_max_distance = Mat(max_distance, Range(ymin, ymax), Range(xmin, xmax));	 //max_distance中为遍历的都是０，比有值的小，要去掉
		double minVal = 0; //最大值一定要赋初值，否则运行时会报错
		Point minLoc;
		minMaxLoc(cut_max_distance, &minVal, NULL, &minLoc, NULL);
		oval_a = minVal;
		oval_center = minLoc + Point(xmin, ymin);				//实际坐标补偿
	}
}

int imgDetection::hough_ellipse(vector<Point> contour)
{
	double G, XX, YY;
	int B;
	Mat hough_space(floor(oval_a + 1), 180, CV_8UC1, Scalar(0));  //高度:a，宽度180

	for (int k = 0; k<contour.size(); k++)
	{
		for (int w = 0; w<180; w++)
		{
			G = w * CV_PI / 180; //角度转换为弧度
			XX = pow(((contour.at(k).y - oval_center.y)*cos(G) - (contour.at(k).x - oval_center.x)*sin(G)), 2) / (oval_a*oval_a);
			YY = pow(((contour.at(k).y - oval_center.y)*sin(G) + (contour.at(k).x - oval_center.x)*cos(G)), 2);

			B = floor(sqrt(abs(YY / (1 - XX))) + 1);
			if (B>0 && B <= oval_a)
			{
				hough_space.at<uchar>(B, w) += 1;
			}
		}
	}
	double Circumference;
	double maxVal = 0; //最大值一定要赋初值，否则运行时会报错
	Point maxLoc;
	minMaxLoc(hough_space, NULL, &maxVal, NULL, &maxLoc);
	oval_b = maxLoc.y;
	oval_theta = maxLoc.x;
	Circumference = 2 * CV_PI*oval_b + 4 * (oval_a - oval_b);

	return maxVal;
}

Mat imgDetection::draw_Eliipse(Mat src)
{
	cout << "长轴：" << oval_a << endl;
	cout << "短轴：" << oval_b << endl;
	cout << "椭圆中心：" << oval_center << endl;
	cout << "oval_theta：" << oval_theta << endl;
	ellipse(src, oval_center, Size(oval_b, oval_a), oval_theta, 0, 360, Scalar(255, 255, 255), 3);
	return  src;
}

void imgDetection::getNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, float & depth, bool &state)
{
	cv::Mat whiteImg;
	getHSVwhite(_img, whiteImg, numDirect);
	std::vector<cv::Rect> resultRect = findrect(whiteImg, numDirect);//先找白色区域

	Mat imageROI;
	state = false;
	std::vector<cv::Point> depthDetectArea;
	std::vector<cv::Rect>::iterator it;
	for (it = resultRect.begin(); it != resultRect.end(); it++)
	{
		imageROI = _img(*it); //切出数字区域
		getRoiNum(imageROI, numDirect, wantedNum, center, depthDetectArea, state);//对标志牌区域识别数字
		if (state == true)
		{
			// 正确时计算深度，并且getRoiNum()里用小图得到的中心值需要偏移
			std::vector<cv::Point>::iterator it1;
			for (it1 = depthDetectArea.begin(); it1 != depthDetectArea.end(); it1++)
			{
				it1->x += it->x;
				it1->y += it->y;
			}
			center.x += it->x;
			center.y += it->y;

			//cout << "center" << center << endl;

			getStandBoradDepth(_img, depthDetectArea, depth);
			break;
		}
	}

}

void imgDetection::getHSVwhite(cv::Mat &src, cv::Mat &dst, int numDirect)
{
	//Mat imageROI;
	//imageROI = _img(Rect(0, _img.rows / 2, _img.cols, _img.rows / 2)); //切出下半部分

	/*  白色  */

	int iLowH = 0;
	int iHighH = 180;

	int iLowS = 0;
	int iHighS = 10;

	int iLowV = 40;
	int iHighV = 255;

	if (numDirect == NumDirection::LYING)
	{
		iLowV = 40;
		iHighV = 255;
	}
	else if (numDirect == NumDirection::STANDING)
	{
		iLowV = 150;
		iHighV = 255;
	}

	Mat imgHSV;
	cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										  //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  

	cv::inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), dst); //Threshold the image 

																						   //cv::imshow("img", imgThresholded);
																						   //cv::waitKey(0);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(dst, dst, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("imgThresholded", dst);
	//cv::waitKey(0);

}

vector<cv::Rect> imgDetection::findrect(cv::Mat &_img, int numDirect)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 

	vector<cv::Rect> resultRect;
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 1.4, minRatio = 0.88, ratio = double(width) / height;
		int maxH = 250, minH = 20, maxW = 280, minW = 20;

		if (numDirect == NumDirection::LYING)
		{
			maxH = 250;
			minH = 20;
			maxW = 280;
			minW = 20;
			maxRatio = 1.4;
			minRatio = 0.88;
		}
		else if (numDirect == NumDirection::STANDING)
		{
			maxH = 120;
			minH = 20;
			maxW = 200;
			minW = 30;
			maxRatio = 1.8;
			minRatio = 1.4;
		}
		//
		//cout << height << ',' << width << ',' << ratio << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height > maxH || width > maxW)  continue;
		if (height < minH || width < minW)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;

		resultRect.push_back(rect);
		//cout << "found white!" << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);
	}

	return resultRect;
}

void imgDetection::getRoiNum(cv::Mat &_img, int numDirect, int wantedNum, cv::Point &center, std::vector<cv::Point> &depthDetectArea, bool &state)
{
	//Mat imageROI;
	//imageROI = _img(Rect(0, _img.rows / 2, _img.cols, _img.rows / 2)); //切出下半部分

	Mat gray_img, thresholded_img, final_img;//stand用二值化,lying用hsv

											 /*  黑色  */

	int iLowH = 0;
	int iHighH = 180;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 100;

	if (numDirect == NumDirection::LYING)
	{

		Mat imgHSV;
		cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV    
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), thresholded_img); //Threshold the image 
		thresholded_img = ~thresholded_img;
	}
	else if (numDirect == NumDirection::STANDING)
	{

		// 灰度化
		cvtColor(_img, gray_img, COLOR_RGB2GRAY);

		// 二值化
		threshold(gray_img, thresholded_img, 150, 255, THRESH_BINARY);
		//cv::imshow("binary_img", binary_img);
		//cv::waitKey(0);
		//cout << binary_img.type() << endl;
	}

	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	morphologyEx(thresholded_img, thresholded_img, MORPH_CLOSE, element);

	cv::Mat rgb_thresholded_img;
	cvtColor(thresholded_img, rgb_thresholded_img, COLOR_GRAY2RGB);

	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("thresholded_img", thresholded_img);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	//findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	findContours(thresholded_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point());

	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = HASNOT;

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 0.8, minRatio = 0.4, ratio = double(width) / height;
		int maxH = 250, minH = 10, maxW = 280, minW = 10;

		if (numDirect == NumDirection::LYING)
		{
			maxH = 250;
			minH = 10;
			maxW = 280;
			minW = 10;
			maxRatio = 0.8;
			minRatio = 0.4;
		}
		else if (numDirect == NumDirection::STANDING)
		{
			maxH = 80;
			minH = 12;
			maxW = 60;
			minW = 7;
			maxRatio = 0.7;
			minRatio = 0.4;
		}
		//
		//cout << height << ',' << width << ',' << ratio << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height > maxH || width > maxW)  continue;
		if (height < minH || width < minW)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;

		//cout << height << ',' << width << ',' << ratio << endl;

		drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		std::vector<cv::Point> vertex;
		vertex.push_back(cv::Point(px + 1, py + 1));
		vertex.push_back(cv::Point(px + width - 2, py + 1));
		vertex.push_back(cv::Point(px + width - 2, py + height - 2));
		vertex.push_back(cv::Point(px + 1, py + height - 2));

		Mat dst;
		fullScreenNum(rgb_thresholded_img, dst, vertex);

		//cv::imshow("dst", dst);
		//cv::waitKey(0);

		int predNum;
		float distance;
		detectNum(dst, predNum, distance); //模型位置在imgDectction.h里改
										   //cout << "predNum" << predNum << endl;
										   //cout << "distance:" << distance << endl;

		if (distance < 4e9)
			cout << predNum << endl;

		if (predNum == wantedNum
			&& distance < 4e9) //KNN距离阈值
		{

			center.x = px + width / 2;
			center.y = py + height / 2;
			state = HAS;
			//cout << predNum << "," << wantedNum << endl;

			//circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
			//cv::imshow("showImg", showImg);
			//cv::waitKey(1);

			depthDetectArea.push_back(center);
			depthDetectArea.push_back(cv::Point(px, py));
			depthDetectArea.push_back(cv::Point(px + width, py + height));

		}

	}

}

void imgDetection::getStandBoradDepth(cv::Mat &_img, std::vector<cv::Point> P, float &boardDepth)
{

	/*  白色  */

	int iLowH = 0;
	int iHighH = 180;

	int iLowS = 0;
	int iHighS = 10;

	int iLowV = 150;
	int iHighV = 255;

	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)


	int boradMin_y = imgThresholded.rows, boradMin_x = imgThresholded.cols, boradMax_y = 0, boradMax_x = 0;
	bool finishFlag = false;

	int offset = 3;   //为防止数字框外还有黑点，设置的偏移量
					  //向下
	for (int i = P[2].y + offset; i < imgThresholded.rows; i++)
	{
		//获取第 i行首像素指针 
		uchar * p = imgThresholded.ptr<uchar>(i);
		//对第i 行的每个像素(byte)操作 
		if (p[P[0].x] == 0)
		{
			boradMax_y = i;
			break;
		}
	}

	//向上
	for (int i = P[1].y - offset; i >= 0; i--)
	{
		//获取第 i行首像素指针 
		uchar * p = imgThresholded.ptr<uchar>(i);
		//对第i 行的每个像素(byte)操作 
		if (p[P[0].x] == 0)
		{
			boradMin_y = i;
			break;
		}
	}

	//向右

	for (int j = P[2].x + offset; j < imgThresholded.cols; j++)
	{
		for (int i = P[1].y; i <= P[2].y; i++)
		{
			uchar * p = imgThresholded.ptr<uchar>(i);
			if (p[j] == 0)
			{
				boradMax_x = j;
				finishFlag = true;
				break;
			}
		}
		if (finishFlag == true) break;
	}

	finishFlag = false;
	for (int j = P[1].x - offset; j >= 0; j--)
	{
		for (int i = P[1].y; i <= P[2].y; i++)
		{
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

	if (boradMin_y == imgThresholded.rows || boradMin_x == imgThresholded.cols || boradMax_y == 0 || boradMax_x == 0)
	{
		boardDepth = -1;
		printf("牌子在边缘，深度估计不准！已置-1\n");
		return;
	}

	Eigen::Matrix3f K;
	K << 269.5, 0, 319.5,
		0, 269.5, 239.5,
		0, 0, 1;

	Eigen::Vector3f X;
	X << (float)(boradMax_y - boradMin_y), (float)(boradMax_x - boradMin_x), 0.0f;

	Eigen::RowVector3f Y;
	Y = K.inverse()*X;

	boardDepth = (0.7 / Y[0] + 1.3 / Y[1]) / 2;
	cout << "boardDepth" << boardDepth << endl;

}


void imgDetection::getLyingBorad(cv::Mat &_img, cv::Point &center, bool &state)
{

	state = HASNOT;
	//Mat imageROI;
	//imageROI = _img(Rect(0, _img.rows / 2, _img.cols, _img.rows / 2)); //切出下半部分

	/*  黑色  */

	int iLowH = 0;
	int iHighH = 180;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 20;

	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = HASNOT;

	int maxpy = 0;
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float stdRatio = 1.72, bias = 0.3, ratio = double(height) / width;

		//cout << height << ',' << width << ',' << ratio << endl;



		if (height > 30 || width > 200)  continue;
		//if (height < 12 || width < 12)  continue;
		if (width < 40)  continue;
		//	if ((ratio < stdRatio - bias) || (ratio > stdRatio + bias))	 continue;
		if (ratio > 0.3)	 continue;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
		////cv::imshow("img", imageContours);
		////cv::waitKey(0);

		box[i] = minAreaRect(contours[i]);
		std::vector<cv::Point2f> perspective_points(4);
		cv::Point2f rects[4];
		box[i].points(rects);

		if (maxpy < py)
		{
			maxpy = py;
			center = Point(px + width / 2, py + height / 2);
		}

		state = HAS;
		Mat showImg = _img.clone();
		circle(showImg, center, 4, Scalar(255, 255, 255), -1, 8, 0);


	}

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

void imgDetection::detectAruco(cv::Mat &A, cv::Mat &depthImg, int &rightMarkerIds, std::ofstream &_outfile, bool &state)
{
	/*Mat HSA;
	cvtColor(A, HSA, COLOR_BGR2HSV);
	int iLowH = 0 / 2;
	int iHighH = 360 / 2;

	int iLowS = 0 * 255 / 100;
	int iHighS = 34 * 255 / 100;

	int iLowV = 0 * 255 / 100;
	int iHighV = 100 * 255 / 100;
	Mat imgThresholded;

	inRange(HSA, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
	Mat grayA;
	cvtColor(A, grayA, COLOR_RGB2GRAY);
	for (int i = 0; i < imgThresholded.rows; ++i)
	{
	//获取第 i行首像素指针
	uchar * p = imgThresholded.ptr<uchar>(i);
	//对第i 行的每个像素(byte)操作
	for (int j = 0; j < imgThresholded.cols; ++j)
	if (p[j]<200)grayA.at<uchar>(i, j) = 255;
	}
	for (int i = 0; i < grayA.rows; ++i)
	{
	//获取第 i行首像素指针
	uchar * p = grayA.ptr<uchar>(i);
	//对第i 行的每个像素(byte)操作
	for (int j = 0; j < grayA.cols; ++j)
	{
	if (p[j] >100)grayA.at<uchar>(i, j) = 255;
	if (p[j] < 100)grayA.at<uchar>(i, j) = 0;
	}
	}*/

	//Mat B = A.clone();

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
	cv::aruco::detectMarkers(thA, dictionary, markerCorners, markerIds, params, rejectedCandidates);

	if (markerIds.size() != 0) {
		rightMarkerIds = markerIds[0];
		printf("有二维码:%d\n", rightMarkerIds);
	}
	else
	{
		printf("没有任何二维码\n");
		return;
	}
	vector<cv::Point2f> rightmarkerCorners = markerCorners[0];
	int maxMarkerSize = 0;

	//用最大对角距离筛选多个二维码
	if (markerCorners.size() > 0)
	{
		for (int i = 0; i <= markerCorners.size() - 1; i++)
		{
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

	if (depthImg.at<float>(rightMarkerCenter) >= 254.0f)
	{
		printf("二维码不在有深度的柱子上\n");
		state = false;
		return;
	}


	//float * p = imgThresholded.ptr<float>(i);
	////对第i 行的每个像素(byte)操作 
	//for (int j = 0; j < imgThresholded.cols; ++j)
	//	if (p[j]<200)grayA.at<float>(i, j) = 255;
	for (auto iter = alreadyMarkerIds.begin(); iter != alreadyMarkerIds.end(); iter++)
	{
		if (rightMarkerIds == *iter)
		{
			printf("已经记录过了，溜了");
			state = true;
			return;
		}
	}
	//检验二维码的值是否对应要求的值
	int i = 0;
	for (auto iter = arucoIds.begin(); iter != arucoIds.end(); iter++)
	{
		i++;
		if (rightMarkerIds == *iter)
		{
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

void imgDetection::detectAruco(cv::Mat &A, int &rightMarkerIds, std::ofstream &_outfile, bool &state)
{
	/*Mat HSA;
	cvtColor(A, HSA, COLOR_BGR2HSV);
	int iLowH = 0 / 2;
	int iHighH = 360 / 2;

	int iLowS = 0 * 255 / 100;
	int iHighS = 34 * 255 / 100;

	int iLowV = 0 * 255 / 100;
	int iHighV = 100 * 255 / 100;
	Mat imgThresholded;

	inRange(HSA, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
	Mat grayA;
	cvtColor(A, grayA, COLOR_RGB2GRAY);
	for (int i = 0; i < imgThresholded.rows; ++i)
	{
	//获取第 i行首像素指针
	uchar * p = imgThresholded.ptr<uchar>(i);
	//对第i 行的每个像素(byte)操作
	for (int j = 0; j < imgThresholded.cols; ++j)
	if (p[j]<200)grayA.at<uchar>(i, j) = 255;
	}
	for (int i = 0; i < grayA.rows; ++i)
	{
	//获取第 i行首像素指针
	uchar * p = grayA.ptr<uchar>(i);
	//对第i 行的每个像素(byte)操作
	for (int j = 0; j < grayA.cols; ++j)
	{
	if (p[j] >100)grayA.at<uchar>(i, j) = 255;
	if (p[j] < 100)grayA.at<uchar>(i, j) = 0;
	}
	}*/

	//Mat B = A.clone();

	state = false;

	Mat B;
	Mat thA;
	cvtColor(A, B, COLOR_BGR2GRAY);
	equalizeHist(B, thA);
	vector< int > markerIds;
	vector< vector<Point2f> > markerCorners, rejectedCandidates;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
	params->adaptiveThreshWinSizeMin = 5;
	params->adaptiveThreshWinSizeMax = 50;
	params->polygonalApproxAccuracyRate = 0.15;
	params->minMarkerPerimeterRate = 0.008;
	cv::aruco::detectMarkers(thA, dictionary, markerCorners, markerIds, params, rejectedCandidates);

	if (markerIds.size() != 0) {
		rightMarkerIds = markerIds[0];
		printf("有二维码:%d\n", rightMarkerIds);
	}
	else
	{
		printf("没有任何二维码\n");
		return;
	}
	vector<cv::Point2f> rightmarkerCorners = markerCorners[0];
	int maxMarkerSize = 0;

	//用最大对角距离筛选多个二维码
	if (markerCorners.size() > 0)
	{
		for (int i = 0; i <= markerCorners.size() - 1; i++)
		{
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

	//float * p = imgThresholded.ptr<float>(i);
	////对第i 行的每个像素(byte)操作 
	//for (int j = 0; j < imgThresholded.cols; ++j)
	//	if (p[j]<200)grayA.at<float>(i, j) = 255;
	for (auto iter = alreadyMarkerIds.begin(); iter != alreadyMarkerIds.end(); iter++)
	{
		if (rightMarkerIds == *iter)
		{
			printf("已经记录过了，溜了");
			state = true;
			return;
		}
	}
	//检验二维码的值是否对应要求的值
	int i = 0;
	for (auto iter = arucoIds.begin(); iter != arucoIds.end(); iter++)
	{
		i++;
		if (rightMarkerIds == *iter)
		{
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

//fang jin bao


// 提取白色区域
//前视停机坪只可能出现在图像的下半部分，所以可以将上半部分先删除
Mat imgDetection::image_init_fd(Mat src_img)
{
	Mat gray_img, binary_img;
	// 灰度化
	cvtColor(src_img, gray_img, COLOR_RGB2GRAY);

	//blur(gray_img, gray_img, Size(2, 2));
	//imshow("【去噪后】", gray_img);

	// 二值化
	threshold(gray_img, binary_img, black_thresh_fd, 255, THRESH_BINARY);
	//imshow("二值化", binary_img);

	//canny边缘提取
	//Canny(gray_img, binary_img, 80, 100);

	//Mat dst(binary_img.cols, binary_img.rows, CV_32FC1, Scalar(0));
	//src_img.copyTo(dst, binary_img)

	//Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	//膨胀腐蚀//膨胀操作
	//dilate(binary_img, binary_img, element);

	// 开闭操作
	//morphologyEx(binary_img, binary_img, MORPH_OPEN, element);
	//morphologyEx(binary_img, binary_img, MORPH_CLOSE, element);
	//imshow("开闭操作", binary_img);		
	Mat half_binary_img = binary_img(Rect(0, binary_img.rows / 2, binary_img.cols, binary_img.rows / 2));
	//imshow("out", out);
	return half_binary_img;
}

std::vector<Rect> imgDetection::findrect_fd(Mat half_binary_img)
{
	std::vector<Rect> feature_rect;

	std::vector<std::vector<Point>> contours;		//寻找轮廓
	findContours(half_binary_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	double area_whole = 0.7;	//定义最小的连通域面积与矩形框面积之比

	for (int i = 0; i < contours.size(); i++)
	{
		Rect black_rect = boundingRect(contours[i]);
		double area = contourArea(contours[i]);
		//矩形框不能紧挨着图像边缘，符合一定长宽比
		if (black_rect.x > 2 && black_rect.y > 2 && black_rect.x + black_rect.width < half_binary_img.cols && black_rect.y + black_rect.height < half_binary_img.rows)
		{
			if (area / (black_rect.width*black_rect.height) > area_whole && 1.0*black_rect.width / black_rect.height > 3.0)
			{
				feature_rect.push_back(black_rect);
				//rectangle(binary_img, rect, Scalar(0), 1);
				//cout << contourArea(contours[i]) << " " << black_rect.width << " " << black_rect.height <<" "<< contourArea(contours[i]) / (black_rect.width*black_rect.height) << endl;
			}
		}
	}
	return feature_rect;
}

std::map<int, image_params_fd> imgDetection::get_num_roi_almost_fd(std::vector<Rect> feature_rect, Mat src_img)
{
	std::map<int, image_params_fd> params;
	for (int i = 0; i < feature_rect.size(); i++)
	{
		Rect roi_rect;
		roi_rect.x = feature_rect[i].x - feature_rect[i].width / 2.0;
		roi_rect.y = feature_rect[i].y - feature_rect[i].width*2.0 / 3.0 + src_img.rows / 2.0;
		roi_rect.width = feature_rect[i].width * 2;
		roi_rect.height = feature_rect[i].width*2.0 / 3.0;

		if (roi_rect.width + roi_rect.x > 640) roi_rect.width = 640 - roi_rect.x;
		if (roi_rect.height + roi_rect.y > 480) roi_rect.height = 480 - roi_rect.y;

		struct image_params_fd ipfd = {
			Point2f(roi_rect.x + roi_rect.width / 2.0,roi_rect.y + roi_rect.height / 2),
			src_img(roi_rect)
		};
		params.insert(pair<int, image_params_fd>(roi_rect.x, ipfd));
		//imshow("almost", src_img(roi_rect));
		//waitKey();
	}
	return params;
}

//main part!!!!
//策略：二值化后，找图像角点，然后将图像反变换到矩形，框出数字
Mat imgDetection::num_roi_process_fd(Mat num_roi_almost)
{
	Mat Number;
	//将数字的roi区域二值化，切除多余部分
	//二值化
	Mat binary_num_roi;
	cvtColor(num_roi_almost, binary_num_roi, COLOR_RGB2GRAY);
	threshold(binary_num_roi, binary_num_roi, white_thresh_fd, 255, THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));	//第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
	erode(binary_num_roi, binary_num_roi, element);					//腐蚀操作

																	//开闭操作
																	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
																	//morphologyEx(num_roi_almost, num_roi_almost, MORPH_OPEN, element);
																	//morphologyEx(num_roi_almost, num_roi_almost, MORPH_CLOSE, element);

																	//连通区域涂白
	std::vector<vector<Point>> dp_contours;
	findContours(binary_num_roi, dp_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < dp_contours.size(); i++)
	{
		drawContours(binary_num_roi, dp_contours, i, Scalar(255), FILLED);

	}
	if (dp_contours.size() > 1)
		return Number;

	//imshow("bin", binary_num_roi);
	//waitKey();
	//

	//寻找角点
	std::vector<Point2f> conners;//检测到的角点
	int maxConers = 4;//检测角点上限
	double qualityLevel = 0.1;//最小特征值
	double minDistance = 10;//最小距离

	goodFeaturesToTrack(binary_num_roi, conners, maxConers, qualityLevel, minDistance);
	//绘制角点
	//for (int i = 0; i < conners.size(); i++)
	//{
	//	circle(num_roi_almost, conners[i], 3, Scalar(255 & rand(), 255 & rand(), 255 & rand()), 2, 8, 0);
	//}

	if (conners.size() < 4)
		return Number;

	Point2f rect_point[4];

	float index_num[4];
	for (int count, i = 0; i < 4; i++)
	{
		count = 0;
		for (int j = 0; j < 4; j++)
		{
			if (i == j)
				continue;
			if (conners[i].y == conners[j].y)
				if (i < j)
					count++;
			if (conners[i].y > conners[j].y)
				count++;
		}
		index_num[count] = i;
	}
	if (conners[index_num[0]].x > conners[index_num[1]].x)
	{
		rect_point[0] = conners[index_num[1]];
		rect_point[1] = conners[index_num[0]];
	}
	else
	{
		rect_point[0] = conners[index_num[0]];
		rect_point[1] = conners[index_num[1]];
	}
	if (conners[index_num[2]].x > conners[index_num[3]].x)
	{
		rect_point[2] = conners[index_num[3]];
		rect_point[3] = conners[index_num[2]];
	}
	else
	{
		rect_point[2] = conners[index_num[2]];
		rect_point[3] = conners[index_num[3]];
	}

	Point2f pts_dst[4] = { Point2f(0,0),Point2f(120,0), Point2f(0,80),Point2f(120,80) };
	Mat trans = getPerspectiveTransform(rect_point, pts_dst);
	Mat warp;
	warpPerspective(num_roi_almost, warp, trans, Size(120, 80));
	//imshow("warp", warp);
	//waitKey();

	//imshow("角点检测", num_roi_almost);
	//waitKey();

	//矫正后的图像
	cvtColor(warp, warp, COLOR_RGB2GRAY);
	threshold(warp, warp, white_thresh_fd, 255, THRESH_BINARY);
	//寻找数字轮廓
	std::vector<vector<Point>> num_contours;
	findContours(warp, num_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < num_contours.size(); i++) {
		Rect rect = boundingRect(num_contours[i]);
		if (1.0 * rect.width / rect.height<1.0 && 1.0 * rect.width / rect.height>0.3 && rect.width*rect.height>num_lowest_fd && (1.0*rect.width*rect.height / (num_roi_almost.cols*num_roi_almost.rows)<0.5))
		{
			//cout << 1.0*rect.width*rect.height / (num_roi_almost.cols*num_roi_almost.rows) << endl;
			//cout << rect.width << " " << rect.height << endl;
			//rectangle(num_roi, rect, Scalar(0), 1);			// 绘制数字矩形
			Number = warp(rect);
			//imshow("数字轮廓", Number);
			//waitKey();
			break;
		}
	}
	return Number;
}

int imgDetection::num_predict_fd(Mat num_roi, Ptr<KNearest> num_knn)
{
	int K = 3;	// num_knn->getDefaultK;
	Mat img_temp;

	resize(num_roi, img_temp, Size(image_cols_fd, image_rows_fd), (0, 0), (0, 0), INTER_AREA);//统一图像
	threshold(img_temp, img_temp, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("【roi】", img_temp);
	Mat sample_mat(1, image_cols_fd*image_rows_fd, CV_32FC1);
	for (int i = 0; i < image_rows_fd*image_cols_fd; ++i)
		sample_mat.at<float>(0, i) = (float)img_temp.at<uchar>(i / image_cols_fd, i % image_cols_fd);
	Mat matResults;//保存测试结果
	num_knn->findNearest(sample_mat, K, matResults);//knn分类预测    
													//cout << matResults << endl;
	return (int)matResults.at<float>(0, 0);
}

void imgDetection::get_fd_util(cv::Mat &_img, int wantedNum, cv::Point &center, bool &state)
{
	map<int, Point2f> lyingNumResult;

	lyingNumResult = get_fd_result(_img);

	int maxY = 0;
	map<int, Point2f>::iterator it;
	for (it = lyingNumResult.begin(); it != lyingNumResult.end(); it++) {
		if (it->first == wantedNum && it->second.y > maxY) {  //输出与需求数字对应，且在最下边的结果
			state = true;
			center = it->second;
		}
	}

}

std::map<int, Point2f> imgDetection::get_fd_result(Mat src_img)
{
	//map<int, ellipse_params> num_ellipse;
	std::map<int, Point2f> num_rect;

	Mat half_binary_img = image_init_fd(src_img).clone();
	std::vector<Rect> feature_rect = findrect_fd(half_binary_img);
	if (!feature_rect.empty())
	{
		std::map<int, image_params_fd> params = get_num_roi_almost_fd(feature_rect, src_img);
		std::map<int, image_params_fd>::iterator iter;
		//map<int, Point2f> num_rect;
		for (iter = params.begin(); iter != params.end(); iter++)
		{
			Mat num_roi = num_roi_process_fd(iter->second.roi);
			if (!num_roi.empty())
			{
				//cv::imshow("num_roi", num_roi);
				//cv::waitKey(0);

				int result = num_predict_fd(num_roi, num_knn);
				/*cout << result << endl;*/
				num_rect.insert(pair<int, Point2f>(result, iter->second.center));
			}

		}
	}

	//imshow("binary_img", half_binary_img);
	//waitKey();

	return num_rect;
}

void imgDetection::getCircleDown(cv::Mat &_img, cv::Point &center, bool &state)
{
	/*  红色  */

	int iLowH1 = 156;
	int iHighH1 = 180;
	int iLowH2 = 0;
	int iHighH2 = 10;

	int iLowS = 43;
	int iHighS = 255;

	int iLowV = 46;
	int iHighV = 255;


	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	Mat imgThresholded, imgThresholded1, imgThresholded2;
	inRange(imgHSV, Scalar(iLowH1, iLowS, iLowV), Scalar(iHighH1, iHighS, iHighV), imgThresholded1); //Threshold the image 

																									 //cv::imshow("img", imgThresholded);
																									 //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded1, imgThresholded1, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	inRange(imgHSV, Scalar(iLowH2, iLowS, iLowV), Scalar(iHighH2, iHighS, iHighV), imgThresholded2); //Threshold the image 

																									 //cv::imshow("img", imgThresholded);
																									 //cv::waitKey(0);
	morphologyEx(imgThresholded2, imgThresholded2, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	imgThresholded = imgThresholded1 | imgThresholded2;//  红色分布在两个hsv区域内，分别筛选后合起来

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = false;

	int minY = _img.rows - 1;   //选择最上面的红色部分
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 20, minRatio = 1.7, ratio = double(width) / height;

		//cout << height << ',' << width << ',' << ratio << endl;

		drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height > 100 || width > 250)  continue;
		if (height < 5 || width < 30)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;

		//cout << height << ',' << width << ',' << ratio << endl;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		cv::Point tempCenter = cv::Point(px + width / 2, py + height / 2);

		state = true;
		//cout << predNum << "," << wantedNum << endl;

		if (minY > tempCenter.y)
		{
			minY = tempCenter.y;
			center = tempCenter;
		}
	}

	if (state == true)
	{
		//cv::Mat showImg = _img.clone();
		//circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
		//cv::imshow("showImg", showImg);
		//cv::waitKey(0);
	}

}

void imgDetection::blueDetection(cv::Mat &_img, bool &state, WallLocation &wallLocation)
{
	/*  蓝色  */

	int iLowH = 100;
	int iHighH = 124;

	int iLowS = 43;
	int iHighS = 255;

	int iLowV = 20;
	int iHighV = 255;


	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	//morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	for (int i = 0; i < 2; ++i)
	{
		uchar* iData = imgThresholded.ptr<uchar>(i);
		for (int j = 0; j < imgThresholded.cols; ++j)	iData[j] = 0;
	}

	for (int i = imgThresholded.rows - 2; i < imgThresholded.rows; ++i)
	{
		uchar* iData = imgThresholded.ptr<uchar>(i);
		for (int j = 0; j < imgThresholded.cols; ++j)	iData[j] = 0;
	}

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = false;

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 8, minRatio = 1, ratio = double(width) / height;

		//cout << height << ',' << width << ',' << ratio << endl;

		//cout << "num" << contours.size() << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height < 100 || width < 20)  continue;

		//cout << height << ',' << width << ',' << ratio << endl;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		if (rect.x + rect.width / 2 < _img.cols / 2)		wallLocation = WallLocation::LEFT_WALL;
		else											wallLocation = WallLocation::RIGHT_WALL;

		state = true;
		return;
		//cout << predNum << "," << wantedNum << endl;

		//cv::Mat showImg = _img.clone();
		//circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
		//cv::imshow("showImg", showImg);
		//cv::waitKey(0);
	}

}

void imgDetection::getTreeTop(cv::Mat &_img, cv::Point &center, bool &state)      //俯视看树桩
{

	/*  土黄色  */

	int iLowH = 7;
	int iHighH = 30;

	int iLowS = 17;
	int iHighS = 122;

	int iLowV = 106;
	int iHighV = 255;


	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  

	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = false;

	int minDist = _img.rows*_img.rows + _img.cols*_img.cols;   //选择最上面的红色部分
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 1.2, minRatio = 0.8, ratio = double(width) / height;

		//cout << height << ',' << width << ',' << ratio << endl;
		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height > 250 || width > 250)  continue;
		if (height < 50 || width < 50)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;

		//cout << height << ',' << width << ',' << ratio << endl;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 0, 0), 3, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		cv::Point tempCenter = cv::Point(px + width / 2, py + height / 2);
		int tempDist = sqrt(pow((tempCenter.x - _img.cols / 2), 2) + pow((tempCenter.y - _img.rows / 2), 2));

		state = true;

		if (minDist > tempDist)
		{
			minDist = tempDist;
			center = tempCenter;
		}
	}

	//if (state == true)
	//{
	//	cv::Mat showImg = _img.clone();
	//	circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
	//	cv::imshow("showImg", showImg);
	//	cv::waitKey(0);
	//}

}


void imgDetection::getWallTop(cv::Mat &_img, bool &state, int &centerX)      //屋顶墙中心
{
	/*  屋顶蓝色  */

	int iLowH = 100;
	int iHighH = 109;

	int iLowS = 70;
	int iHighS = 185;

	int iLowV = 22;
	int iHighV = 100;


	Mat imgHSV;

	cvtColor(_img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
										   //cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);


	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	//imshow("Thresholded Image2", imgThresholded); //show the thresholded image
	//闭操作 (连接一些连通域)

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //最小外接矩形画布 
	state = false;

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		int width, height, px, py;
		px = rect.x;
		py = rect.y;
		width = rect.width;
		height = rect.height;

		float maxRatio = 100, minRatio = 1.5, ratio = double(width) / height;

		//cout << height << ',' << width << ',' << ratio << endl;

		//drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);

		if (height < 50 || width < 500)  continue;
		if ((ratio < minRatio) || (ratio > maxRatio))	 continue;


		//cout << height << ',' << width << ',' << ratio << endl;
		drawContours(imageContours, contours, i, cv::Scalar(255, 255, 255), -1, 8, hierarchy);
		//cv::imshow("img", imageContours);
		//cv::waitKey(0);

		cvtColor(imageContours, imageContours, COLOR_BGR2GRAY); //Convert the captured frame from BGR to HSV
																//imshow("imgHSV", imgHSV);

		std::vector<int> leftX, rightX;
		float yratio = 0.4; //取上半部分的比例
		int bias = 150;
		int leftRow = imageContours.rows, rightRow = imageContours.rows;

		bool stopFlag = false;
		for (int i = py; i <= int(py + height*yratio); ++i)
		{
			float* iData = imageContours.ptr<float>(i);
			for (int j = px + width / 2; j <= px + width; ++j)
			{
				if (iData[j] > 1.0f && j <= imageContours.cols / 2 + bias && j >= imageContours.cols / 2 - bias)
				{
					leftX.push_back(j);
					leftRow = i;
					stopFlag = true;
				}
			}
			if (stopFlag == true) break;
		}


		stopFlag = false;
		for (int i = py; i <= int(py + height*yratio); ++i)
		{
			float* iData = imageContours.ptr<float>(i);
			for (int j = px + width / 2; j >= px; --j)
			{
				if (iData[j] > 1.0f && j <= imageContours.cols / 2 + bias && j >= imageContours.cols / 2 - bias)
				{
					rightX.push_back(j);
					rightRow = i;
					stopFlag = true;
				}
			}
			if (stopFlag == true) break;
		}

		//cout << "rightX.size()" << rightX.size() << endl;
		//cout << "leftX.size()" << leftX.size() << endl;
		int sum = 0;
		//for (int i = 0; i < leftX.size(); i++)
		//{
		//	sum += (leftX[i] + rightX[i]);
		//}
		//centerX = sum / (leftX.size() * 2);

		if (leftRow < rightRow)
		{
			for (int i = 0; i < leftX.size(); i++)
			{
				sum += leftX[i];
			}
			centerX = sum / leftX.size();
		}

		else
		{
			for (int i = 0; i < rightX.size(); i++)
			{
				sum += rightX[i];
			}
			centerX = sum / rightX.size();
		}

		Mat showImg = _img.clone();
		line(showImg, cv::Point(centerX,0), cv::Point(centerX, showImg.cols), Scalar(255), 1);
		cv::imshow("showImg", showImg);
		cv::waitKey(0);

		state = true;

	}
}