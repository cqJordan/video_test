// VideoSyn.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "synopsis.h"
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;
void getFgImagedd(const IplImage * frame, IplImage * fgImg, const IplImage * bgImg, const int threshold) {
	if (frame == NULL || fgImg == NULL || bgImg == NULL) {
		printf("Fail: There exists NULL input. Fail to get Foreground Image!\n");
		return;
	}

	if (frame->nChannels != 3 || fgImg->nChannels != 3 || bgImg->nChannels != 3) {
		printf("Fail: All input image should be color!\nframe channel:%d\nfgImg channel:%d\nbgImg channel:%d\n", \
			frame->nChannels, fgImg->nChannels, bgImg->nChannels);
		return;
	}

	CvMat * frameMat = cvCreateMat(frame->height, frame->width, CV_32FC3);
	CvMat * fgMat = cvCreateMat(fgImg->height, fgImg->width, CV_32FC3);
	CvMat * bgMat = cvCreateMat(bgImg->height, bgImg->width, CV_32FC3);

	cvConvert(frame, frameMat);
	cvConvert(fgImg, fgMat);
	cvConvert(bgImg, bgMat);

	cvSmooth(frameMat, frameMat, CV_GAUSSIAN, 5, 5, 4, 4); //高斯滤波先，以平滑图像

	cvAbsDiff(frameMat, bgMat, fgMat); //当前帧跟背景图相减(求背景差并取绝对值)

	cvThreshold(fgMat, fgImg, threshold, 255, CV_THRESH_BINARY); //二值化前景图(这里采用特定阈值进行二值化)

																 //进行形态学滤波，去掉噪音
	cvErode(fgImg, fgImg, 0, 1);
	cvDilate(fgImg, fgImg, 0, 1);

	//释放矩阵内存
	cvReleaseMat(&frameMat);
	cvReleaseMat(&fgMat);
	cvReleaseMat(&bgMat);
}

int _tmain(int argc, _TCHAR* argv[])
{

	const char file_path[] = "test.mp4";
	const char file_out_path[] = "test_dav_out.avi";

	int fps, frame_number;
	CvSize size;
	IplImage* bgImage = NULL;
	const int frame_num_used = 500;
	list< list<tube *> > database;
	int code = (int)CV_FOURCC('d', 'i', 'v', '3');//MPEG-4.3 codec
	bgModeling(file_path, file_out_path, frame_num_used, &bgImage, fps, frame_number, size, code);

	code = (int)CV_FOURCC('x', 'v', 'i', 'd');

	cvShowImage("平均背景", bgImage);
	cvWaitKey();
	Mat frame, bgimg, fgmask,pp;
	Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2();
	VideoCapture cap("test.dav");
	CvCapture* pCapture = cvCaptureFromFile("test.dav");//自己选取一段avi视频
	IplImage *fgImg = cvCreateImage(cvSize(bgImage->width, bgImage->height), bgImage->depth, bgImage->nChannels);
	IplImage *frame1 = NULL;
	IplImage imgTmp ;
	IplImage *changgeBG =NULL,*img1=NULL;
	for (;;) {
		cap >> frame;
		imshow("原始图像帧", frame);
		mog->apply(frame, fgmask, 0.01);//-1默认
		imshow("opencv自带前景", fgmask);
		mog->getBackgroundImage(bgimg);
		imshow("opencv自带背景", bgimg);
		//getFgImagedd(const IplImage * frame, IplImage * fgImg, const IplImage * bgImg, const int threshold) {
		frame1= cvQueryFrame(pCapture);

		imgTmp = bgimg;
		changgeBG = cvCloneImage(&imgTmp);

		getFgImagedd(frame1, fgImg, changgeBG, 30); //帧差法获取前景图像
		img1 = cvCreateImage(cvGetSize(fgImg), IPL_DEPTH_8U, 1);//创建目标图像  
		cvCvtColor(fgImg, img1, CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)  
		//pp = cvarrToMat(fgImg);
		//imshow("帧差法获取前景图像",pp);
		cvShowImage("帧差法获取前景彩色图像", fgImg);
		cvShowImage("帧差法获取前景灰度图像", img1);
		if (waitKey(30) >= 0) break;
	}
	waitKey();
	buildTrackDB(file_path, bgImage, database);
	mergeDB(database, file_out_path, fps, size, bgImage);
	return 0;
}

