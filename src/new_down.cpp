#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string.h>
#include "opencv2/ml.hpp"
#include "new_down.h"
#include <opencv2/opencv.hpp>  


using namespace std;
using namespace cv;
using namespace ml;


Mat down_image_init(Mat src_image)
{
	Mat gray_image, binary_image;

	//�ҶȻ�
	cvtColor(src_image, gray_image, COLOR_BGR2GRAY);

	//ȥ��
	//blur(gray_image, gray_image, Size(4, 4));
	//imshow("��ȥ���", gray_image);

	//sobel�˲�
	/*
	Mat dx, dy;
	Sobel(gray_image, dx, CV_8U, 1, 0, 3, 1, 0);
	convertScaleAbs(dx, dx);
	Sobel(gray_image, dy, CV_8U, 0, 1, 3, 1, 0);
	convertScaleAbs(dy, dy);
	addWeighted(dx, 1.0, dy, 1.0, 0, gray_image);
	imshow("��sobel�˲���", gray_image);
	*/

	//��ֵ��
	threshold(gray_image, binary_image, 120, 255, THRESH_BINARY);

	//��������
	Mat bin_ero;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(binary_image, bin_ero, element);
	//morphologyEx(binary_image, binary_image, MORPH_OPEN, element);
	//morphologyEx(binary_image, binary_image, MORPH_CLOSE, element);
	//imshow("���������㡿", binary_image);

	//ȡ��
	//binary_image = 255 - binary_image;
	//imshow("new_down", binary_image);
	//waitKey(1);
	return binary_image;
}

Mat down_image_hsv_init(Mat src_image)
{
	//int iLowH = 0;
	//int iHighH = 86;

	//int iLowS = 0;
	//int iHighS = 147;

	//int iLowV = 24;
	//int iHighV = 255;
	int iLowH = 0;
	int iHighH = 43;

	int iLowS = 0;
	int iHighS = 138;

	int iLowV = 176;
	int iHighV = 255;
	Mat imgHSV, imgThresholded;

	cvtColor(src_image, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
												//cvtColor(imageROI, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	cv::imshow("src_image", src_image);
	cv::waitKey(0);
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image 

																								  //cv::imshow("img", imgThresholded);
																								  //cv::waitKey(0);

																								  //��������
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	//Mat bin_ero;
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//erode(imgThresholded, bin_ero, element);
	//morphologyEx(binary_image, binary_image, MORPH_OPEN, element);
	//morphologyEx(binary_image, binary_image, MORPH_CLOSE, element);
	//imshow("���������㡿", binary_image);

	//ȡ��
	//binary_image = 255 - binary_image;
	//imshow("new_down", binary_image);
	//waitKey(1);
	//cv::imshow("img", imgThresholded);
	//cv::waitKey(0);

	return imgThresholded;
}

map<int, image_params> image_split(Mat binary_image)
{
	Mat binary_image_ref = binary_image.clone();
	Size image_size(binary_image.size());

	vector<vector<Point>> contours;
	findContours(binary_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	map<int, image_params> num_roi_almost;
	for (int i = 0; i < contours.size(); i++) {
		//cv::Mat imageContours = cv::Mat::zeros(binary_image.size(), CV_32FC3); //��С��Ӿ��λ��� 
		//drawContours(imageContours, contours, i, Scalar(255,255,255), 1); // ��������
		//cv::imshow("imageContours", imageContours);
		//cv::waitKey(0);


		RotatedRect rect = minAreaRect(contours[i]);

		if (rect.size.height / (rect.size.width + 0.01)<0.7 || rect.size.height / (rect.size.width + 0.01)>1.3 || rect.size.height * rect.size.width<lowest_num)
			continue;
		//if(rect.center.x-rect.size.width/2<0.01|| rect.center.x + rect.size.width / 2>binary_image.cols-0.01|| rect.center.y - rect.size.width / 2<0.01 || rect.center.x + rect.size.width / 2>binary_image.cols - 0.01)
		//ɾ��������ͼ��߽��ϵľ���
		Point2f vertices[4];
		rect.points(vertices);
		int flag = 0;
		//for (int j = 0; j < 4; j++) {
		//	//	line(binary_image, vertices[j], vertices[(j + 1) % 4], Scalar(255), 1);
		//	if (vertices[j].x<0.01 || vertices[j].y<0.01 || vertices[j].x>binary_image.cols - 0.01 || vertices[j].y>binary_image.rows - 0.01)
		//	{
		//		flag = 1;
		//		break;
		//	}
		//}
		//if (flag == 1)
		//{
		//	flag = 0;
		//	continue;
		//}
		//cout << "h:" << rect.size.height << " w:" << rect.size.width << endl;
		//����ת����ת����������
		float rotateAngle = 0;
		if (rect.angle < -45) rotateAngle = rect.angle + 90;
		else if (rect.angle > 45) rotateAngle = rect.angle - 90;
		else rotateAngle = rect.angle;

		Mat rotation = getRotationMatrix2D(rect.center, rotateAngle, 1.0);
		Mat rot_img;
		warpAffine(binary_image_ref, rot_img, rotation, image_size);
		//imshow("����ת��", rot_img);
		//waitKey();

		//cout << rect.center.x << " " << rect.center.y << " " << rect.size.width << " " << rect.size.height << endl;

		int imgx = rect.center.x - (rect.size.width / 2.0) > 0 ? (int)(rect.center.x - (rect.size.width / 2.0)) : 0;
		int imgy = rect.center.y - (rect.size.height / 2.0) > 0 ? (int)(rect.center.y - (rect.size.height / 2.0)) : 0;
		int imgwidth = (imgx + rect.size.width) < rot_img.cols ? (int)(rect.size.width) : (int)(rot_img.cols - imgx);
		int imgheight = (imgy + rect.size.height) < rot_img.rows ? (int)(rect.size.height) : (int)(rot_img.rows - imgy);
		Mat roi = rot_img(Rect(imgx, imgy, imgwidth, imgheight));
		//��ͼ�����
		//�Ȱѳ���Ϊ���̱�Ϊ��
		//if (roi.cols < roi.rows)
		//{
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//}

		//float x = 0.0f;		// roi.cols / 2.0;
		float y = 0.0f;		// roi.rows / 2.0;
		vector<vector<Point>> num_contours;
		findContours(roi, num_contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		//ԭ��ȡƫ�����ģ�����ѡ��ȡƽ�������Խ��10������
		//vector<Point2f> point_temp;
		for (int k = 0; k < num_contours.size(); k++)
		{
			Rect num_rect = boundingRect(num_contours[k]);
			//if (num_rect.x<2 || num_rect.y<2 || num_rect.x + num_rect.width>roi.cols - 2 || num_rect.y + num_rect.height>roi.rows - 2)
			//	break;
			int flag = 0;

			//x += num_rect.x + num_rect.width / 2.0f;
			y += num_rect.y + num_rect.height / 2.0f;
			/*
			if (fabsf(num_rect.x + num_rect.width / 2.0 - roi.cols / 2.0) > fabsf(x - roi.cols / 2.0))
			x = num_rect.x + num_rect.width / 2.0;
			if (fabsf(num_rect.y + num_rect.height / 2.0 - roi.rows / 2.0) > fabsf(y - roi.rows / 2.0))
			y = num_rect.y + num_rect.height / 2.0;*/

			//rectangle(roi, num_rect, Scalar(0,0,255), 1);
		}
		//x /= num_contours.size();
		y /= num_contours.size();
		//cout << x << " " << roi.cols << " " << y << " " << roi.rows << endl;

		//if (y < roi.rows / 2.0)
		//{
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//	transpose(roi, roi);
		//	flip(roi, roi, 1);
		//}
		/*if (fabsf(x - roi.cols / 2.0) > fabsf(y - roi.rows / 2.0))
		{
		if (x > roi.cols / 2.0)
		{
		transpose(roi, roi);
		flip(roi, roi, 1);
		}
		else
		{
		transpose(roi, roi);
		flip(roi, roi, 0);
		}
		}
		else
		{
		if (y < roi.rows / 2.0)
		{
		transpose(roi, roi);
		flip(roi, roi, 1);
		transpose(roi, roi);
		flip(roi, roi, 1);
		}
		}*/

		struct image_params ip {
			rect.center,
				roi
		};

		num_roi_almost.insert(pair<int, image_params>(fabsf(rect.center.x - roi.cols), ip));
	}
	imshow("���ָ�ͼ��", binary_image);
	waitKey(0);
	return num_roi_almost;
}

Mat down_num_roi_process(Mat num_roi_almost)
{
	//Ѱ����������
	vector<vector<Point>> num_contours;
	findContours(num_roi_almost, num_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat Number;
	Mat temp;
	//int j = 0;

	for (int i = 0; i < num_contours.size(); i++) {
		Rect rect = boundingRect(num_contours[i]);
		if (1.0 * rect.width / (rect.height + 0.01)<0.9 && 1.0 * rect.width / (rect.height + 0.01)>0.3 && rect.width*rect.height>100)
		{
			//cout << rect.width << " " << rect.height << endl;
			//rectangle(num_roi, rect, Scalar(0), 1);			// �������־���
			Number = num_roi_almost(rect);
			//j++;
			//imshow("down��������", Number);
			//waitKey(1);
			break;
		}
	}
	//if (j > 1) 
	//	return temp;
	//else 
	return Number;
	//j = 0;
}

//ʶ��
int down_num_predict(Mat num_rect, Ptr<KNearest> num_knn)
{
	const int K = 3;	//testModel->getDefaultK()  
	Mat img_temp;
	resize(num_rect, img_temp, Size(image_cols_num, image_rows_num), (0, 0), (0, 0), INTER_AREA);//ͳһͼ��
	threshold(img_temp, img_temp, 0, 255, THRESH_BINARY | THRESH_OTSU);

	//imshow("��roi��", img_temp);
	//waitKey();

	Mat sample_mat(1, image_cols_num*image_rows_num, CV_32FC1);
	for (int i = 0; i < image_cols_num*image_rows_num; ++i)
		sample_mat.at<float>(0, i) = (float)img_temp.at<uchar>(i / image_cols_num, i % image_cols_num);
	Mat matResults;//������Խ��
	num_knn->findNearest(sample_mat, K, matResults);//knn����Ԥ��    
													//cout << matResults << endl;
	return (int)matResults.at<float>(0, 0);
}

void getTreeTop(cv::Mat &_img, cv::Point &center, bool &state)      //���ӿ���׮
{

	/*  ����ɫ  */

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
	//�ղ��� (����һЩ��ͨ��)

	//cv::imshow("imgThresholded", imgThresholded);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box(contours.size());

	cv::Mat imageContours = cv::Mat::zeros(_img.size(), CV_32FC3); //��С��Ӿ��λ��� 
	state = false;

	int minDist = _img.rows*_img.rows + _img.cols*_img.cols;   //ѡ��������ĺ�ɫ����
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

	if (state == true)
	{
		//cv::Mat showImg = _img.clone();
		//circle(showImg, center, 8, Scalar(0, 0, 255), -1, 8, 0);
		//cv::imshow("showImg", showImg);
		//cv::waitKey(1);
	}

}


map<int, Point2f> get_down_result(Mat src_image, Ptr<KNearest> num_knn)
{
	Mat binary_image = down_image_hsv_init(src_image).clone();

	map<int, image_params> num_roi_almost = image_split(binary_image);
	// knnʶ��  
	map<int, image_params>::iterator iter;
	map<int, Point2f> num_rect;

	//cout << num_roi_almost.size() << endl;
	//static int image_name = 0;
	for (iter = num_roi_almost.begin(); iter != num_roi_almost.end(); iter++)
	{
		imshow("test", iter->second.roi);
		waitKey();
		Mat roi = down_num_roi_process(iter->second.roi);
		//cout << 1 << endl;
		if (roi.empty())
		{

			//cout << 1 << endl;
			//num_rect.insert(pair<int, Point2f>(10, iter->second.center));
		}
		else
		{
			//cout << 2 << endl;
			int result = down_num_predict(roi, num_knn);
			cout << "down:" << result << endl;
			num_rect.insert(pair<int, Point2f>(result, iter->second.center));
			//imwrite("dataset/" + to_string(image_name) + ".jpg", roi);
			//image_name++;


		}

	}
	return num_rect;
}

void distCorrect(cv::Mat &src, cv::Mat &dst)
{

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	//�ڲξ���, ���㸴�ƴ��룬Ҳ��Ҫ���ҵĲ���������ͷ����һ��...
	cameraMatrix.at<double>(0, 0) = 533.6104383276903;
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(0, 2) = 314.7377259417282;
	cameraMatrix.at<double>(1, 1) = 530.391280459703;
	cameraMatrix.at<double>(1, 2) = 263.2418306925699;
	//�����������Ҫ���ҵĲ���~
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

}