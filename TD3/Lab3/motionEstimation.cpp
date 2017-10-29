#include "motionEstimation.hpp"

#include "OpticalFlow.h"

void computeMotionEstimation(const cv::Mat &prev,
			     const cv::Mat &next,
			     cv::Mat &motionVectors)
{
  //We need to convert data
  cv::Mat dprev, dnext;
  prev.convertTo(dprev, CV_64FC1);
  next.convertTo(dnext, CV_64FC1);

  DImage Im1(dprev.cols, dprev.rows, 1);
  DImage Im2(dprev.cols, dprev.rows, 1);

  int offset=0;
  for(int y = 0; y < dprev.rows; y += 1) {
    for(int x = 0; x < dprev.cols; x += 1){
      Im1[offset] = dprev.at<double>(y, x)/255.0;
      Im2[offset] = dnext.at<double>(y, x)/255.0;
      ++offset;
    }
  }

  
  DImage vx, vy, warpI2, mflow;
#if 1
  const double alpha= 0.012;
  const double ratio=0.75;
  const int minWidth= 20;
  const int nOuterFPIterations = 7;
  const int nInnerFPIterations = 1;
  const int nSORIterations= 30;
#else
  const double alpha= 0.012;
  const double ratio=0.75;
  const int minWidth= 40; //20;
  const int nOuterFPIterations = 3; //7;
  const int nInnerFPIterations = 1;
  const int nSORIterations= 40; //30;
#endif
  
  OpticalFlow::Coarse2FineFlow(vx, vy, warpI2, Im2, Im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);


  
  motionVectors = cv::Mat(prev.rows, prev.cols, CV_32FC2);

  double *ptrx = vx.data();
  double *ptry = vy.data();
  for(int y = 0; y < vx.height(); y += 1){
    for(int x = 0; x < vx.width(); x += 1){
      motionVectors.at<cv::Point2f>(y, x)=cv::Point2f((float)ptrx[y*vx.width()+x], (float)ptry[y*vx.width()+x]);
    }
  }
  
}
