#ifndef BLOCKMATCHING_HPP
#define BLOCKMATCHING_HPP

#include <opencv2/core/core.hpp>

/*
  Full-search block-matching mono-resolution method.

  Compute motionVectors between m1 and m2 images.
  Consider blocks of size blockSize.
  Search matching blocks in window of size windowSize x windowSize, 
  centered around current block.

  Output matrix motionVectors will be of type CV_32SC2.
 */
extern void blockMatchingMono(const cv::Mat &m1, const cv::Mat &m2,
			      int blockSize,
			      int windowSize,
			      cv::Mat &motionVectors);

/*
  Full-search block-matching multi-resolution method.

  Compute motionVectors between m1 and m2 images with a pyramid of nbLevels levels.
  Consider blocks of size blockSize at highest level.
  Search matching blocks in window of size windowSize x windowSize, 
  centered around current block, at highest level.

  Output images and motionVectors at each level.
  motionVectors[0] is the motion vector matrix at the highest level.
  Each output motion vector matrix will be of type CV_32SC2.
  levels1[i] is the image m1 at level i.
 */
extern void blockMatchingMulti(const cv::Mat &m1, const cv::Mat &m2,
			       int blockSize,
			       int windowSize,
			       int nbLevels,
			       std::vector<cv::Mat> &levels1,			       
			       std::vector<cv::Mat> &levels2,			       
			       std::vector<cv::Mat> &motionVectors);


#endif /* ! BLOCKMATCHING_HPP */
