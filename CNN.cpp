// CNN.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include <iostream>
#include <algorithm>
#include <cstdlib>	
#include <chrono>
#include <opencv2/opencv.hpp>
#include "kernal.h"
#pragma optimize(3)
using namespace cv;
using namespace std;

float dst[3 * 128 * 128];

void CHW_convert()
{
	Mat img = imread("kun.jpg");
	int pos;
	uchar* p_uchar;
	imshow("face-classification", img);
	for (int h = 0; h < img.rows; ++h)
	{
		p_uchar = img.data + h * img.step;
		for (int w = 0; w < img.cols; ++w)
		{
			pos = 128 * h + w;
			dst[pos] = p_uchar[2] / 255.0f;                 //R
			dst[128 * 128 + pos] = p_uchar[1] / 255.0f;     //G
			dst[128 * 128 * 2 + pos] = p_uchar[0] / 255.0f; //B
			p_uchar += 3;
		}
	}
}
int main()
{
	auto start = chrono::steady_clock::now();
	CHW_convert();//调整为 CHW 格式
	
	First_conv(128);
	ReLU(first_out, FIR_O_SIZE);
	MaxPool(first_out, firstPool, 64, 16);


	Second_conv(32);
	ReLU(second_out, SEC_O_SIZE);
	MaxPool(second_out, secondPool, 32, 32);
	
	
	Third_conv(16);
	ReLU(third_out,THID_O_SIZE);

	Gemm();
	softmax();

	auto end = chrono::steady_clock::now();
	cout << "用时：" <<chrono::duration_cast<chrono::milliseconds>(end - start).count() << "毫秒 " << endl;
    waitKey(0);
}
