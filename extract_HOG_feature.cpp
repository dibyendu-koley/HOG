/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "extract_HOG_feature.h"
//#include"opencv2/opencv.hpp"
/*@list file name and store in vector*/
vector<string> listFile(const char* dir_name){
DIR *dpdf;
struct dirent *epdf;
vector<string> ImgFileName;
dpdf = opendir(dir_name);
if (dpdf != NULL){
   while (epdf = readdir(dpdf)){
        if( strstr( epdf->d_name, ".jpg" ) ){
                ImgFileName.push_back(epdf->d_name);
        }
   }
}
return (ImgFileName);
}


extract_HOG_feature::extract_HOG_feature(void)
{
//YCrCb threshold
// You can change the values and see what happens
Y_MIN  = 0;
Y_MAX  = 255;
Cr_MIN = 133;
Cr_MAX = 173;
Cb_MIN = 77;
Cb_MAX = 127;
}
extract_HOG_feature::~extract_HOG_feature(void)
{
}
void extract_HOG_feature::extract_NEG()
{
     String FullFileName;
 char SaveHogDesFileName[100] = "Negative.xml";
 int FileNum=96;
//----------------------------------dibyendu
const char* dir_name = "/home/dibyendu/Desktop/Desktop_upto_20_07_2016/opencv/negative_images/";
vector<string> ImgFileName;
ImgFileName =  listFile(dir_name);
//----------------------------------dibyendu
 vector< vector < float> > v_descriptorsValues;
 vector< vector < Point> > v_locations;

for(vector<string>::const_iterator i = ImgFileName.begin(); i != ImgFileName.end(); ++i) {
cout << dir_name+*i << "\n";
  //read image file
  Mat img, img_gray;
  img = imread(dir_name+*i);
  
  //resizing
  resize(img, img, Size(64,48) ); //Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
  //gray
  cvtColor(img, img_gray, CV_RGB2GRAY);

  //extract feature
  HOGDescriptor d( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
  vector< float> descriptorsValues;
  vector< Point> locations;
  d.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);

  //printf("descriptor number =%d\n", descriptorsValues.size() );
  v_descriptorsValues.push_back( descriptorsValues );
  v_locations.push_back( locations );
  //show image
  imshow("origin", img);

  waitKey(5);
 }

 //save to xml
 FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE); //FileStorage::READ
 //2d vector to Mat
 int row=v_descriptorsValues.size(), col=v_descriptorsValues[0].size();
 printf("col=%d, row=%d\n", row, col);
 Mat M(row,col,CV_32F);
 //save Mat to XML
 for(int i=0; i< row; ++i)  
  memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptorsValues[i].data(),col*sizeof(float));
 //write xml
 write(hogXml, "Descriptor_of_images",  M);

 hogXml.release();
}
void extract_HOG_feature::extract_POS()
{
    //variables
 String FullFileName;
 char SaveHogDesFileName[100] = "Positive.xml";
 int FileNum=96;
//----------------------------------dibyendu
const char* dir_name = "/home/dibyendu/Desktop/Desktop_upto_20_07_2016/opencv/car/13-04-2016-car-dekho-2-3-lkh-kolkata/imageclipper/";
vector<string> ImgFileName;
ImgFileName =  listFile(dir_name);
//----------------------------------dibyendu
 vector< vector < float> > v_descriptorsValues;
 vector< vector < Point> > v_locations;

for(vector<string>::const_iterator i = ImgFileName.begin(); i != ImgFileName.end(); ++i) {
cout << dir_name+*i << "\n";
  //read image file
  Mat img, img_gray;
  img = imread(dir_name+*i);
  
  //resizing
  resize(img, img, Size(64,48) ); //Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
  //gray
  cvtColor(img, img_gray, CV_RGB2GRAY);

  //extract feature
  HOGDescriptor d( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
  vector< float> descriptorsValues;
  vector< Point> locations;
  d.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);

  //printf("descriptor number =%d\n", descriptorsValues.size() );
  v_descriptorsValues.push_back( descriptorsValues );
  v_locations.push_back( locations );
  //show image
  imshow("origin", img);

  waitKey(5);
 }

 //save to xml
 FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE); //FileStorage::READ
 //2d vector to Mat
 int row=v_descriptorsValues.size(), col=v_descriptorsValues[0].size();
 printf("col=%d, row=%d\n", row, col);
 Mat M(row,col,CV_32F);
 //save Mat to XML
 for(int i=0; i< row; ++i)  
  memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptorsValues[i].data(),col*sizeof(float));
 //write xml
 write(hogXml, "Descriptor_of_images",  M);
 hogXml.release();
}
void extract_HOG_feature::train_HOG_SVM()
{
    
 //Read Hog feature from XML file
 ///////////////////////////////////////////////////////////////////////////
 printf("1. Feature data xml load\n");
 //create xml to read
 FileStorage read_PositiveXml("Positive.xml", FileStorage::READ);
 FileStorage read_NegativeXml("Negative.xml", FileStorage::READ);

 //Positive Mat
 Mat pMat;
 read_PositiveXml["Descriptor_of_images"] >> pMat;
 //Read Row, Cols
 int pRow,pCol;
 pRow = pMat.rows; pCol = pMat.cols;

 //Negative Mat
 Mat nMat;
 read_NegativeXml["Descriptor_of_images"] >> nMat;
 //Read Row, Cols
 int nRow,nCol;
 nRow = nMat.rows; nCol = nMat.cols;

 //Rows, Cols printf
 printf("   pRow=%d pCol=%d, nRow=%d nCol=%d\n", pRow, pCol, nRow, nCol );
 //release
 read_PositiveXml.release();
 //release
 read_NegativeXml.release();
 /////////////////////////////////////////////////////////////////////////////////

 //Make training data for SVM
 /////////////////////////////////////////////////////////////////////////////////
 printf("2. Make training data for SVM\n");
 //descriptor data set
 Mat PN_Descriptor_mtx( pRow + nRow, pCol, CV_32FC1 ); //in here pCol and nCol is descriptor number, so two value must be same;
 memcpy(PN_Descriptor_mtx.data, pMat.data, sizeof(float) * pMat.cols * pMat.rows );
 int startP = sizeof(float) * pMat.cols * pMat.rows;
 memcpy(&(PN_Descriptor_mtx.data[ startP ]), nMat.data, sizeof(float) * nMat.cols * nMat.rows );
 //data labeling
 Mat labels( pRow + nRow, 1, CV_32FC1, Scalar(-1.0) );
    labels.rowRange( 0, pRow ) = Scalar( 1.0 );
 /////////////////////////////////////////////////////////////////////////////////

 //Set svm parameter
 /////////////////////////////////////////////////////////////////////////////////
 printf("4. SVM training\n");
 CvSVM svm;
 CvSVMParams params;
 params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 10000, 1e-6 );
 /////////////////////////////////////////////////////////////////////////////////

 //Training
 /////////////////////////////////////////////////////////////////////////////////
 svm.train(PN_Descriptor_mtx, labels, Mat(), Mat(), params);

 //Trained data save
 /////////////////////////////////////////////////////////////////////////////////
 printf("5. SVM xml save\n");
 svm.save( "trainedSVM.xml" );
 
// FileStorage hogXml("testXML.xml", FileStorage::WRITE); //FileStorage::READ
// write(hogXml, "Data", PN_Descriptor_mtx);
// write(hogXml, "Label", labels);
// hogXml.release();
}
void extract_HOG_feature::test_HOG_SVM(const char* dir_name1)
{
    

 //variables
 //char FullFileName[100];
 //char FirstFileName[100]="./images/upperbody"; //"./NegaImages/Negative";      // 
 int FileNum=96; //262;
 String FullFileName;
 char const* ca;
const char* dir_name = dir_name1;

 //char SaveHogDesFileName[100] = "Positive.xml";
 const char* SaveHogDesFileName = "trainedSVM.xml";
 //const char* SaveHogDesFileName = argv[2];

 //Load trained SVM xml data
 CvSVM svm;
 //svm.load("trainedSVM.xml");
 svm.load(SaveHogDesFileName);
vector<string> ImgFileName;
ImgFileName =  listFile(dir_name);

 //count variable
 int nnn=0, ppp=0;
for(vector<string>::const_iterator i = ImgFileName.begin(); i != ImgFileName.end(); ++i) {
// for(int i=0; i< FileNum; ++i) {
cout << dir_name+*i << "\n";
//  sprintf_s(FullFileName, "%s%d.png", FirstFileName, i+1);

  //printf("%s\n", FullFileName);

  //read image file
  Mat img, img_gray;
  //img = imread(FullFileName);
  img = imread(dir_name+*i);
  
  //resizing
  //resize(img, img, Size(16,8) ); //Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
  resize(img, img, Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
  //gray
  cvtColor(img, img_gray, CV_RGB2GRAY);

  //Extract HogFeature
  HOGDescriptor d( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
  vector< float> descriptorsValues;
  vector< Point> locations;
  d.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);
  //vector to Mat
  Mat fm = Mat(descriptorsValues);
  
  //Classification whether data is positive or negative
  int result = svm.predict(fm);
  //printf("%s - > %d\n", FullFileName, result);
  FullFileName = dir_name+*i;
  ca = FullFileName.c_str();
  printf("%s - > %d\n", ca, result);

  //Count data
  if(result == 1)
   ppp++;
  else
   nnn++;

  //show image
  imshow("origin", img);

  waitKey(5);
 }

 printf(" positive/negative = (%d/%d) \n", ppp, nnn);
    
}