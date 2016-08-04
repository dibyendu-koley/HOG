/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: root
 *
 * Created on 29 July, 2016, 1:36 PM
 */

#include <cstdlib>
#include "extract_HOG_feature.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
     Mat image;
    image = imread("/home/dibyendu/Desktop/car/car/test/3.jpg", CV_LOAD_IMAGE_COLOR); 
    
    std::vector<Rect> rois;
    Mat roi;
    const char* test_image_dir = argv[1];
    extract_HOG_feature test;
    //test.extract_NEG();
    //test.extract_POS();
    //test.train_HOG_SVM();
    //test.test_HOG_SVM(test_image_dir);
    rois = pre_detect(image);
    String wn="wn";
    ostringstream convert;
    int j=0;
    float scale =1;
    for(vector<Rect>::const_iterator i = rois.begin(); i != rois.end(); ++i) {
        
            convert<<j;
            j++;
            rectangle( image, cvPoint(cvRound(i->x*scale), cvRound(i->y*scale)),
                   cvPoint(cvRound((i->x + i->width-1)*scale), cvRound((i->y + i->height-1)*scale)),
                   Scalar( 255, 0, 0 ), 3, 8, 0);
            roi = image(Rect(i->x,i->y+25,i->width,i->height*.4));
            test.test_HOG_SVM_after_preprocess(roi);
        }
    //test.convert_HOG_SVM();
    return 0;
}

