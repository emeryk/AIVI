#ifndef HORNSCHUNCK_HPP
#define HORNSCHUNCK_HPP

#include <opencv2/core/core.hpp>

/*
  C++ Wrapper to legacy cvCalcOpticalFlowHS C API function 

  http://docs.opencv.org/2.4.9/modules/legacy/doc/motion_analysis.html#calcopticalflowhs

  Calculates the optical flow for two images using Horn-Schunck algorithm.

    prev – First image, 8-bit, single-channel
    curr – Second image, 8-bit, single-channel
    use_previous – Flag that specifies whether to use the input velocity as initial approximations or not.
    velx – Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel
    vely – Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel
    lambda – Smoothness weight. The larger it is, the smoother optical flow map you get.
    criteria – Criteria of termination of velocity computing
 */
extern void computeOpticalFlowHornSchunck(const cv::Mat &prev,
					  const cv::Mat &curr,
					  bool use_previous,
					  cv::Mat &velx,
					  cv::Mat &vely,
					  double lambda,
					  cv::TermCriteria criteria);


extern void computeOpticalFlowHornSchunckMono(const cv::Mat &m1, const cv::Mat &m2,
					      double lambda,
					      cv::TermCriteria criteria,
					      cv::Mat &motionVectors);

extern void computeOpticalFlowHornSchunckMulti(const cv::Mat &m1, const cv::Mat &m2,
					       double lambda,
					       cv::TermCriteria criteria,
					       int nbLevels,
					       std::vector<cv::Mat> &levels1,
					       std::vector<cv::Mat> &levels2,
					       std::vector<cv::Mat> &motionVectors);

#endif /* HORNSCHUNCK_HPP */
