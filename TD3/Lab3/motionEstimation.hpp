#ifndef MOTION_ESTIMATION_HPP
#define MOTION_ESTIMATION_HPP

#include <opencv2/core/core.hpp>

/*
  Compute motion estimation backwards between @a prev and @a curr, and fill @a motionVectors.

  @a prev and @a curr must be of type CV_8UC1.
  @a motionVectors will be of type CV_32FC2.
 */

extern void computeMotionEstimation(const cv::Mat &prev,
				    const cv::Mat &curr,
				    cv::Mat &motionVectors);

#endif /* ! MOTION_ESTIMATION_HPP */
