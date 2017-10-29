#ifndef GME_HPP
#define GME_HPP

#include <opencv2/core/core.hpp>

extern void computeGME(const cv::Mat &motionVectors, cv::Mat &motionVectorsGlobal);
  
extern void computeGlobalMotionError(const cv::Mat &motionVectors, const cv::Mat &motionVectorsGlobal, cv::Mat &motionError);

#endif /* GME_HPP */
