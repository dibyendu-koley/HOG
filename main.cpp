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
    const char* test_image_dir = argv[1];
    extract_HOG_feature test;
    //test.extract_NEG();
    //test.extract_POS();
    //test.train_HOG_SVM();
    test.test_HOG_SVM(test_image_dir);
    return 0;
}

