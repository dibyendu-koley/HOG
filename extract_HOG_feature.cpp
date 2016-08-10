/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "extract_HOG_feature.h"


//#include"opencv2/opencv.hpp"
/*@list file name and store in vector*/
std::vector<Rect> pre_detect(Mat image)
{
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    //Mat image;
    std::vector<Rect> rois;
    //image = imread("/home/dibyendu/Desktop/car/2.jpg", CV_LOAD_IMAGE_COLOR); 
    namedWindow( "window1", 1 );   imshow( "window1", image );
 
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade;
    face_cascade.load( "cascade.xml" );
 
    // Detect faces
    std::vector<Rect> faces;
    //face_cascade.detectMultiScale( gray_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20, 40) );
    face_cascade.detectMultiScale( gray_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 20));
    Mat roi;
    String wn="wn";
    //char *intStr;
    ostringstream convert;
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        convert<<i;
         roi = image(faces[i]);
         rois.push_back(faces[i]);
         //extract_HOG_feature::test_HOG_SVM_after_preprocess(image, roi);
         //imshow( wn+convert.str(), roi );
        //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        
        //ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }
    return rois;
    //imshow( "Detected Face", image );
    //waitKey(0);
}
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
//Y_MIN  = 0;
//Y_MAX  = 255;
//Cr_MIN = 133;
//Cr_MAX = 173;
//Cb_MIN = 77;
//Cb_MAX = 127;
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
void extract_HOG_feature::convert_HOG_SVM()
{
    //Set svm parameter
 /////////////////////////////////////////////////////////////////////////////////
// printf("1. SVM set parameter\n");
// CvSVM svm;
// CvSVMParams params;
// params.svm_type = CvSVM::C_SVC;
//    params.kernel_type = CvSVM::LINEAR;
//    params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 10000, 1e-6 );
// /////////////////////////////////////////////////////////////////////////////////
//Read Hog feature from XML file
 ///////////////////////////////////////////////////////////////////////////
// printf("2. trained  xml data  load\n");
// //create xml to read
// FileStorage read_PositiveXml("/root/opencv-cpp/HOG_git/HOG/trainedSVM.xml", FileStorage::READ);
// //FileStorage read_NegativeXml("Negative.xml", FileStorage::READ);
//
// //Positive Mat
// Mat pMat;
// read_PositiveXml["Descriptor_of_images"] >> pMat;
// //Read Row, Cols
// int pRow,pCol;
// pRow = pMat.rows; pCol = pMat.cols;
 //char SaveHogDesFileName[100] = "Positive.xml";
 const char* SaveHogDesFileName = "trainedSVM.xml";
 //const char* SaveHogDesFileName = argv[2];

 //Load trained SVM xml data
 CvSVM svm;
 //svm.load("trainedSVM.xml");
 svm.load(SaveHogDesFileName);
 
 
 
    //make firstly, inherited class to access alpha vector and value
int svmVectorSize = svm.get_support_vector_count();
//int featureSize = pCol;
cout<<"pCol:: "<<" svmVectorSize:: "<<svmVectorSize<<"\n";
////prepare, variables
//
//
//Mat sv = Mat(svmVectorSize, featureSize, CV_32FC1, 0.0);
//Mat alp = Mat(1, svmVectorSize, CV_32FC1, 0.0);
//Mat re = Mat(1, featureSize, CV_32FC1, 0.0);
//Mat re2 = Mat(1, featureSize+1, CV_32FC1, 0.0);
//
//
//
////set value to variables
//for(int i=0; i< svmVectorSize; ++i)
//memcpy( (sv.data + i*featureSize), svm.get_support_vector(i), featureSize*sizeof(float) ); //ok
//
//
//double * alphaArr = svm.get_alpha();
//int alphaCount = svm.get_alpha_count();
//
//for(int i=0; i< svmVectorSize; ++i)
//{
//alp.at< float>(0, i) = (float)alphaArr[i];
////printf("alpha[%d] = %lf \n", i, (float)alphaArr[i] );
//}
//
////cvMatMul(alp, sv, re);
//re = alp * sv;
//
//for(int i=0; i< featureSize; ++i)
//re2.at< float>(0,i) = re.at< float>(0,i) * -1;
//re2.at< float>(0,featureSize) = svm.get_rho();
//
////save to 1d vector to XML format!!
//FileStorage svmSecondXML(SVM_HOGDetectorFile, FileStorage::WRITE);
//svmSecondXML << "SecondSVMd" << re2 ;
//
//svmSecondXML.release();
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

  waitKey();
 }

 printf(" positive/negative = (%d/%d) \n", ppp, nnn);
    
}
void extract_HOG_feature::test_HOG_SVM_after_preprocess(Mat roi)
{
//     char const* ca;
 const char* SaveHogDesFileName = "trainedSVM.xml";
// //Load trained SVM xml data
 CvSVM svm;
// //svm.load("trainedSVM.xml");
 svm.load(SaveHogDesFileName);
// //count variable
 int nnn=0, ppp=0;
//for(vector<string>::const_iterator i = ImgFileName.begin(); i != ImgFileName.end(); ++i) {
//// for(int i=0; i< FileNum; ++i) {
//cout << dir_name+*i << "\n";
//  //read image file
  Mat img, img_gray;
//  //img = imread(FullFileName);
  img = roi;
//    //resizing
  resize(img, img, Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
//  //gray
  cvtColor(img, img_gray, CV_RGB2GRAY);
//  //Extract HogFeature
  HOGDescriptor d( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
  vector< float> descriptorsValues;
  vector< Point> locations;
  d.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);
//  //vector to Mat
  Mat fm = Mat(descriptorsValues);
//    //Classification whether data is positive or negative
  int result = svm.predict(fm);
//  FullFileName = dir_name+*i;
//  ca = FullFileName.c_str();
  printf("%d\n", result);
//  //Count data
  if(result == 1)
   ppp++;
  else
   nnn++;
//  //show image
  imshow("origin", img);
  waitKey();
// }
 //printf(" positive/negative = (%d/%d) \n", ppp, nnn);
}
bool extract_HOG_feature::verifySizes(RotatedRect mr){

    float error=0.4;
    //Spain car plate size: 52x11 aspect 4,7272
    float aspect=4.7272;
    //Set a min and max area. All other patchs are discarded
    int min= 15*aspect*15; // minimum area
    int max= 125*aspect*125; // maximum area
    //Get only patchs that match to a respect ratio.
    float rmin= aspect-aspect*error;
    float rmax= aspect+aspect*error;

    int area= mr.size.height * mr.size.width;
    float r= (float)mr.size.width / (float)mr.size.height;
    if(r<1)
        r= (float)mr.size.height / (float)mr.size.width;

    if(( area < min || area > max ) || ( r < rmin || r > rmax )){
        return false;
    }else{
        return true;
    }

}
void extract_HOG_feature::testPlate(Mat roi,bool showSteps)
{
    //convert image to gray
    Mat img_gray;
    cvtColor(roi, img_gray, CV_BGR2GRAY);
    blur(img_gray, img_gray, Size(5,5));    

    //Finde vertical lines. Car plates have high density of vertical lines
    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    if(showSteps)
        imshow("Sobel", img_sobel);

    //threshold image
    Mat img_threshold;
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    if(showSteps)
    imshow("Threshold", img_threshold);
    //Morphplogic operation close
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3) );
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
    if(showSteps)
        imshow("Close", img_threshold);

    //Find contours of possibles plates
    vector< vector< Point> > contours;
    findContours(img_threshold,
            contours, // a vector of contours
            CV_RETR_EXTERNAL, // retrieve the external contours
            CV_CHAIN_APPROX_NONE); // all pixels of each contours

    //Start to iterate to each contour founded
    vector<vector<Point> >::iterator itc= contours.begin();
    vector<RotatedRect> rects;

    //Remove patch that are no inside limits of aspect ratio and area.    
    while (itc!=contours.end()) {
        //Create bounding rect of object
        RotatedRect mr= minAreaRect(Mat(*itc));
        if( !verifySizes(mr)){
            itc= contours.erase(itc);
        }else{
            ++itc;
            rects.push_back(mr);
        }
    }

    // Draw blue contours on a white image
    cv::Mat result;
    roi.copyTo(result);
    cv::drawContours(result,contours,
            -1, // draw all contours
            cv::Scalar(255,0,0), // in blue
            1); // with a thickness of 1
for(int i=0; i< rects.size(); i++){

        //For better rect cropping for each posible box
        //Make floodfill algorithm because the plate has white background
        //And then we can retrieve more clearly the contour box
        circle(result, rects[i].center, 3, Scalar(0,255,0), -1);
        //get the min size between width and height
        float minSize=(rects[i].size.width < rects[i].size.height)?rects[i].size.width:rects[i].size.height;
        minSize=minSize-minSize*0.5;
        //initialize rand and get 5 points around center for floodfill algorithm
        srand ( time(NULL) );
        //Initialize floodfill parameters and variables
        Mat mask;
        mask.create(roi.rows + 2, roi.cols + 2, CV_8UC1);
        mask= Scalar::all(0);
        int loDiff = 30;
        int upDiff = 30;
        int connectivity = 4;
        int newMaskVal = 255;
        int NumSeeds = 10;
        Rect ccomp;
        int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
        for(int j=0; j<NumSeeds; j++){
            Point seed;
            seed.x=rects[i].center.x+rand()%(int)minSize-(minSize/2);
            seed.y=rects[i].center.y+rand()%(int)minSize-(minSize/2);
            circle(result, seed, 1, Scalar(0,255,255), -1);
            int area = floodFill(roi, mask, seed, Scalar(255,0,0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
        }
        if(showSteps)
            imshow("MASK", mask);
        //cvWaitKey(0);

        //Check new floodfill mask match for a correct patch.
        //Get all points detected for get Minimal rotated Rect
        vector<Point> pointsInterest;
        Mat_<uchar>::iterator itMask= mask.begin<uchar>();
        Mat_<uchar>::iterator end= mask.end<uchar>();
        for( ; itMask!=end; ++itMask)
            if(*itMask==255)
                pointsInterest.push_back(itMask.pos());

        RotatedRect minRect = minAreaRect(pointsInterest);

        if(verifySizes(minRect)){
            // rotated rectangle drawing 
            Point2f rect_points[4]; minRect.points( rect_points );
            for( int j = 0; j < 4; j++ )
                line( result, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255), 1, 8 );    

            //Get rotation matrix
            float r= (float)minRect.size.width / (float)minRect.size.height;
            float angle=minRect.angle;    
            if(r<1)
                angle=90+angle;
            Mat rotmat= getRotationMatrix2D(minRect.center, angle,1);

            //Create and rotate image
            Mat img_rotated;
            warpAffine(roi, img_rotated, rotmat, roi.size(), CV_INTER_CUBIC);

            //Crop image
            Size rect_size=minRect.size;
            if(r < 1)
                swap(rect_size.width, rect_size.height);
            Mat img_crop;
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);
            
            Mat resultResized;
            resultResized.create(33,144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
            //Equalize croped image
            Mat grayResult;
            cvtColor(resultResized, grayResult, CV_BGR2GRAY); 
            blur(grayResult, grayResult, Size(3,3));
//            grayResult=histeq(grayResult);
//            if(saveRegions){ 
//                stringstream ss(stringstream::in | stringstream::out);
//                ss << "tmp/" << filename << "_" << i << ".jpg";
//                imwrite(ss.str(), grayResult);
//            }
//            output.push_back(Plate(grayResult,minRect.boundingRect()));
        }
    }       
    if(showSteps) 
        imshow("Contours", result);

    waitKey(0);
}