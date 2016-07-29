/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   extract_HOG_feature.h
 * Author: root
 *
 * Created on 29 July, 2016, 1:42 PM
 */

#ifndef EXTRACT_HOG_FEATURE_H
#define EXTRACT_HOG_FEATURE_H
#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
using namespace cv;
using namespace std;
#ifdef _DEBUG        
#else        
#pragma comment(lib, "opencv_core247.lib")        
#pragma comment(lib, "opencv_imgproc247.lib")        
#pragma comment(lib, "opencv_objdetect247.lib")        
#pragma comment(lib, "opencv_highgui247.lib")        
#endif 
class extract_HOG_feature
{
public:
extract_HOG_feature(void);
~extract_HOG_feature(void);

void extract_NEG(void);
void extract_POS(void);
void train_HOG_SVM(void);
void test_HOG_SVM(const char* dir_name);
private:
int Y_MIN;
int Y_MAX;
int Cr_MIN;
int Cr_MAX;
int Cb_MIN;
int Cb_MAX;
};
vector<string> listFile(const char* dir_name);
#endif /* EXTRACT_HOG_FEATURE_H */

