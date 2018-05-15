/*
opencv3.2.0  vs2015
HOG+SVM detect
*/
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"

#include <stdio.h>
#include <string.h>
#include <cctype>
#include<iostream>
#include <fstream>

#include <sstream>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <opencv2/gpu/gpu.hpp> 

using namespace cv;
using namespace cv::ml;
using namespace std;

#define FILEPATH  "/home/nvidia/catkin_tx2/src/detect_hog/"
void Detect()
{
	//load parameter
  
	ifstream launchFile;
	string launchFilelocation = string(FILEPATH)+"launch.txt";
	launchFile.open(launchFilelocation);
	string firstLine;
	getline(launchFile, firstLine);
	launchFile.close();
	istringstream ss(firstLine);
	
	string videostreamlocation;
	string resultvideolocation;
	string HOG_SVMtxtlocation;
	string partImglocation;
	int ifImshow;
	int ifOutputResultvideo;
	int ifWritePartImg;
	string partImg_ex;
	int videoorcamera;
	ss >> videostreamlocation;
	ss >> resultvideolocation;
	ss >> HOG_SVMtxtlocation;
	ss >> partImglocation;
	ss >> ifImshow;
	ss >> ifOutputResultvideo;
	ss >> ifWritePartImg;
	ss >> partImg_ex;
	ss >> videoorcamera;
	
	VideoCapture capture(videostreamlocation);
	//if(!videoorcamera)  VideoCapture capture(0); 
	if (!capture.isOpened())  
	{  
	    cout << "error" << endl;  
	    waitKey(0);  
	    return;  
	} 
	int w_kevin = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int h_kevin = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	 
	double rate = capture.get(CV_CAP_PROP_FPS);
	VideoWriter writer(string(FILEPATH) +resultvideolocation, CV_FOURCC('M', 'J', 'P', 'G'), rate, Size(w_kevin, h_kevin), true);
	if (!writer.isOpened())
	{
		printf("write1 error .. \n"); return;
	}

	Mat img,img_to_gpu;
	Mat partImg;
	FILE* f = 0;
	char _filename[1024];
	int partImg_count = 0;

	vector<float> detector;
	ifstream fileIn(string(FILEPATH) + HOG_SVMtxtlocation, ios::in);
	float val = 0.0f;
	double fps = 0.0;
	while (!fileIn.eof())
	{
		fileIn >> val;
		//cout << val<<endl;
		detector.push_back(val);
	}
	fileIn.close();
	//gpu::HOGDescriptor HOG_descriptor_detect_gpu;//gpu
	//HOG_descriptor_detect_gpu = gpu::HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//gpu
	//gpu::GpuMat src_GPU;//gpu

	HOGDescriptor hog(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	//HOG_descriptor_detect_gpu.setSVMDetector(detector);//gpu
	hog.setSVMDetector(detector);
	//cout << "set HOG done" << endl;

	while (capture.isOpened())
	{
		capture >> img;
		if (img.empty()) break;
		fflush(stdout);
		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();
		resize(img, img, Size(w_kevin, h_kevin));
		//cvtColor(img,img_to_gpu,CV_BGR2BGRA);//gpu
		//src_GPU.upload(img_to_gpu);//gpu
		//HOG_descriptor_detect_gpu.detectMultiScale(src_GPU, found,0, Size(8, 8), Size(), 1.05);//gpu
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(0, 0), 1.05, 2);
		t = (double)getTickCount() - t;
		//printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		cout << 1000 / (t*1000. / cv::getTickFrequency()) << endl;
		
		size_t i, j;

		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			//xk20180508
			if (found_filtered.size()>1){        //if(i>0)   if (found_filtered.size()>1)
				string partImgName = string(FILEPATH) + partImglocation + partImg_ex + to_string(partImg_count) + ".jpg";
				partImg_count++;
				//namedWindow(partImgName, 1);
				partImg = img(r);
				if(ifWritePartImg){
				  imwrite(partImgName, partImg);
				}
				
			}
		}

		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
		
		if(ifImshow)  imshow("detector", img);
		if(ifOutputResultvideo)  writer << img;
		int key = waitKey(30);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
		
	}
	if (f)  fclose(f);

	capture.release();
	writer.release();
	return;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "detect_hog");
  ros::NodeHandle n; 

  Detect();
  
  return 0;
}
