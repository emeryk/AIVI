#include "blockmatching.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include "utils.hpp"

#include <chrono> //C++11
#include <iostream>


void blockMatchingMono(const cv::Mat &m1, const cv::Mat &m2,
		       int blockSize,
		       int windowSize,
		       cv::Mat &motionVectors)
{
  const auto t1 = std::chrono::high_resolution_clock::now();
  assert(m1.size() == m2.size()
	 && m1.type() == m2.type()
	 && m1.type() == CV_8UC1);
  assert(blockSize > 0);
  assert(windowSize > 0);
  
  //Here we suppose that m1.cols & m1.rows are multiple of blockSize
  const int blockXCount = m1.cols / blockSize;
  const int blockYCount = m1.rows / blockSize;

  motionVectors.create(blockYCount, blockXCount, CV_32SC2);

  //TODO: fill motionVectors.
  
  //We want to traverse all blocks from the m1 image.
  //For each block b1 in m1, we search the block b2 in m2 image 
  //such as computeMSE(b1, b2) is minimal.

  //To call computeMSE we will need to create a cv::Mat corresponding to each block.
  //You can for example use the rowRange()/colRange() methods of cv::Mat.
  
  //Suppose b1 is at position (x1, y1)
  // and the best b2 (that is the block that minimizes computeMSE(b1, b2)) is at (x2Best, y2Best)
  // then the motion vector is (x2Best-x1, y2Best-y1).
  //We can fill the motionVectors matrix with:
  // motionVectors.at<cv::Vec2i>(i, j) = cv::Vec2i(x2Best-x1, y2Best-y1);


  for (int i=0; i<blockYCount; ++i) {

    for (int j=0; j<blockXCount; ++j) {

      const int x1 = j * blockSize;
      const int y1 = i * blockSize;

      assert(x1+blockSize <= m1.cols);
      assert(y1+blockSize <= m1.rows);
      
      const cv::Mat block1 = m1.rowRange(y1, y1+blockSize).colRange(x1, x1+blockSize);
  
      double E0 = DBL_MAX; //std::numeric_limits<double>::max();
      int x2Best = 0;
      int y2Best = 0;
      
      const int minSearchX = std::max(x1-windowSize/2, 0);
      const int minSearchY = std::max(y1-windowSize/2, 0);
      const int maxSearchX = std::min(x1+windowSize/2, m1.cols)-blockSize;
      const int maxSearchY = std::min(y1+windowSize/2, m1.rows)-blockSize;

      for (int x2=minSearchX; x2<maxSearchX; ++x2) {
	for (int y2=minSearchY; y2<maxSearchY; ++y2) {

	  assert(x2>=0);
	  assert(y2>=0);
	  assert(x2+blockSize<=m2.cols);
	  assert(y2+blockSize<=m2.rows);
	  
	  const cv::Mat block2 = m2.rowRange(y2, y2+blockSize).colRange(x2, x2+blockSize);
	  const double E = computeMSE(block1, block2);
	  //const double E = computeSE(block1, block2);
	  if (E < E0) {
	    x2Best = x2;
	    y2Best = y2;
	    E0 = E;
	  }
	  
	}
      }

      assert(E0 != DBL_MAX);
      assert(x2Best>=0);
      assert(y2Best>=0);
      assert(x2Best+blockSize<=m2.cols);
      assert(y2Best+blockSize<=m2.rows);
	  
      const int dx = x2Best - x1;
      const int dy = y2Best - y1;

      motionVectors.at<cv::Vec2i>(i, j) = cv::Vec2i(dx, dy);
      
    }
  }

  const auto t2 = std::chrono::high_resolution_clock::now();
  std::cerr<<"TIME match1="<<std::chrono::duration<double, std::milli>(t2-t1).count()<<"ms\n";
  
}


/*
  OPTIM: we do not use computeMSE(), but computeSE().
  It should be slightly faster.
 */

#define SQUARE(x) ((x) * (x))

static
long
computeSE(const cv::Mat &im, const cv::Mat &imC)
{
  assert(im.cols == imC.cols && im.rows == imC.rows
	 && im.type() == CV_8UC1 && imC.type() == CV_8UC1);

  long sum = 0;

  int rows = im.rows;
  int cols = im.cols;
  
#if 0
  //OPTIM: matrices are never coninuous in our use case, so we comment-out this test.
  if (im.isContinuous() && imC.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
#endif
  
  for (int y = 0; y < rows; ++y) {
    const unsigned char *p1 = im.ptr<unsigned char>(y);
    const unsigned char *p2 = imC.ptr<unsigned char>(y);
    for(int x = 0; x < cols; ++x) {
      sum += SQUARE((long)(p1[x]) - (long)(p2[x]));  //Warning: do not compute with unsigned chars
    }
  }  
  return sum; //OPTIM: We avoid a conversion to double & a divide here...
}


void blockMatchingMono2(const cv::Mat &m1, const cv::Mat &m2,
			int blockSize,
			int windowSize,
			cv::Mat &motionVectors)
{
  //Here motion vectors are used both as an input and an output.
  //If they are non empty, then they are used to initialize block b2 position.

  //TODO : fill motionVectors

  assert(m1.size() == m2.size()
	 && m1.type() == m2.type()
	 && m1.type() == CV_8UC1);
  assert(blockSize > 0);
  assert(windowSize > 0);
  
  //Here we suppose that m1.cols & m1.rows are multiple of blockSize
  const int blockXCount = m1.cols / blockSize;
  const int blockYCount = m1.rows / blockSize;

  if (motionVectors.cols != blockXCount || motionVectors.rows != blockYCount) {
    //There are no existing motion vectors (to initialize best block coordinates)
    //We create motionVectors initialized to 0.
    motionVectors = cv::Mat::zeros(blockYCount, blockXCount, CV_32SC2);
  }

  for (int i=0; i<blockYCount; ++i) {

    const int y1 = i * blockSize;
    assert(y1+blockSize <= m1.rows);
    
    for (int j=0; j<blockXCount; ++j) {

      const int x1 = j * blockSize;

      assert(x1+blockSize <= m1.cols);
      
      const cv::Mat block1 = m1.rowRange(y1, y1+blockSize).colRange(x1, x1+blockSize);
  
      long E0 = LONG_MAX; //std::numeric_limits<long>::max();
      int x2Best = 0;
      int y2Best = 0;

      //Search window is here centered on (xc, yc).
      // (xc, yc)=(x1+v, y1+v) if we use the motion vector v 
      // (xc, yc)=(x1, y1) if we do not use the motion vectors (as in the mono-resolution case). But as motion vectors are initialized to zero in this case, we also have: (xc, yc)=(x1+v, y1+v)
      const cv::Vec2i &v = motionVectors.at<cv::Vec2i>(i, j);
      const int xc = x1 + v[0];
      const int yc = y1 + v[1];

      const int minSearchX = std::max(xc-windowSize/2, 0);
      const int maxSearchX = std::min(xc+windowSize/2, m1.cols)-blockSize;
      const int minSearchY = std::max(yc-windowSize/2, 0);
      const int maxSearchY = std::min(yc+windowSize/2, m1.rows)-blockSize;

      for (int x2=minSearchX; x2<maxSearchX; ++x2) {
	for (int y2=minSearchY; y2<maxSearchY; ++y2) {

	  assert(x2>=0);
	  assert(y2>=0);
	  assert(x2+blockSize<=m2.cols);
	  assert(y2+blockSize<=m2.rows);
	  
	  const cv::Mat block2 = m2.rowRange(y2, y2+blockSize).colRange(x2, x2+blockSize);
	  const long E = computeSE(block1, block2);
	  if (E < E0) {
	    x2Best = x2;
	    y2Best = y2;
	    E0 = E;
	  }
	  
	}
      }

      assert(E0 != LONG_MAX);
      assert(x2Best>=0);
      assert(y2Best>=0);
      assert(x2Best+blockSize<=m2.cols);
      assert(y2Best+blockSize<=m2.rows);
	  
      const int dx = x2Best - x1;
      const int dy = y2Best - y1;

      motionVectors.at<cv::Vec2i>(i, j) = cv::Vec2i(dx, dy);

    }
  }
}




void up(const cv::Mat &m,
	cv::Mat &m2)
{
  assert(m.type() == CV_32SC2);

  m2 = 2 * m;
}


void blockMatchingMulti(const cv::Mat &m1, const cv::Mat &m2,
			int blockSize,
			int windowSize,
			int nbLevels,
			std::vector<cv::Mat> &levels1,
			std::vector<cv::Mat> &levels2,
			std::vector<cv::Mat> &motionVectors)
{
  assert(m1.size() == m2.size()
	 && m1.type() == m2.type()
	 && m1.type() == CV_8UC1);
  assert(blockSize > 0);
  assert(windowSize > 0);
  assert(nbLevels > 0);

  motionVectors.resize(nbLevels);
  levels1.resize(nbLevels);
  levels2.resize(nbLevels);
  std::vector<int> blockSizes(nbLevels);
  levels1[0] = m1;
  levels2[0] = m2;
  blockSizes[0] = blockSize;
  for (int i=1; i<nbLevels; ++i) { //start from 1
    cv::pyrDown(levels1[i-1], levels1[i], cv::Size(levels1[i-1].cols/2, levels1[i-1].rows/2));
    cv::pyrDown(levels2[i-1], levels2[i], cv::Size(levels2[i-1].cols/2, levels2[i-1].rows/2));
    blockSizes[i] = blockSizes[i-1]/2;
  }

  for (int i=nbLevels-1; i>=0; i--) {

    blockMatchingMono2(levels1[i], levels2[i], blockSizes[i], windowSize, motionVectors[i]); 

    if (i > 0) {
      up(motionVectors[i], motionVectors[i-1]); 
    }
  }
}
