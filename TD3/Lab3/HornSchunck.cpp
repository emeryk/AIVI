#include "HornSchunck.hpp"

#include <opencv2/legacy/legacy.hpp>


void computeOpticalFlowHornSchunck(const cv::Mat &prev,
				   const cv::Mat &curr,
				   bool use_previous,
				   cv::Mat &velx,
				   cv::Mat &vely,
				   double lambda,
				   cv::TermCriteria criteria)
{
  assert(prev.size() == curr.size());
  assert(velx.size() == prev.size()
	 && vely.size() == prev.size()
	 && velx.type() == vely.type()
	 && vely.type() == CV_32FC1);
  assert(prev.type() == curr.type() && curr.type() == CV_8UC1);
  
  IplImage i_velx, i_vely, i_prev, i_curr;
  i_velx = IplImage(velx);
  i_vely = IplImage(vely);
  i_prev = IplImage(prev);
  i_curr = IplImage(curr);  

  
  cvCalcOpticalFlowHS(&i_curr, &i_prev, use_previous, &i_velx, &i_vely, lambda, criteria);
}

void computeOpticalFlowHornSchunckMonoAux(const cv::Mat &m1, const cv::Mat &m2,
					  double lambda,
					  cv::TermCriteria criteria,
					  cv::Mat &motionVectors)
{
  assert(motionVectors.empty() || motionVectors.type() == CV_32FC2);

  cv::Mat velx, vely;
  if (!motionVectors.empty()) {
    std::vector<cv::Mat> planes(2);
    cv::split(motionVectors, planes);
    velx = planes[0];
    vely = planes[1];
  }
  else {
    velx.create(m1.rows, m1.cols, CV_32FC1);
    vely.create(m1.rows, m1.cols, CV_32FC1);
  }
  /*
    WARNING: we want to compute the motion backwards.
    However, by default OpenCV (cvCalcOpticalFlowHS) computes the motion forward.

    We can not compute the motion forward and then compute the compensated image using (-dx, -dy) because we have floats coordinates. Thus we have to inverse the parameters to cvCalcOpticalFlowHS().
   */
  computeOpticalFlowHornSchunck(m2, m1, false, velx, vely, lambda, criteria);

  cv::Mat arr[2] = {velx, vely};
  cv::merge(arr, 2, motionVectors);
}

void computeOpticalFlowHornSchunckMono(const cv::Mat &m1, const cv::Mat &m2,
				       double lambda,
				       cv::TermCriteria criteria,
				       cv::Mat &motionVectors)
{
  computeOpticalFlowHornSchunckMonoAux(m1, m2, lambda, criteria, motionVectors);
}

void up(const cv::Mat &m, cv::Size size,
	cv::Mat &m2)
{
  assert(!m.empty());
  assert(m.type() == CV_32FC2);
  std::vector<cv::Mat> planes(2);
  cv::split(m, planes);
  cv::Mat velx = planes[0];
  cv::Mat vely = planes[1];
  cv::Mat velx2, vely2;
  cv::resize(velx, velx2, size, cv::INTER_LINEAR);
  cv::resize(vely, vely2, size, cv::INTER_LINEAR);
  velx = 2*velx2;
  vely = 2*vely2;

  cv::Mat arr[2] = {velx, vely};
  cv::merge(arr, 2, m2);
}

void computeOpticalFlowHornSchunckMulti(const cv::Mat &m1, const cv::Mat &m2,
					double lambda,
					cv::TermCriteria criteria,
					int nbLevels,
					std::vector<cv::Mat> &levels1,
					std::vector<cv::Mat> &levels2,
					std::vector<cv::Mat> &motionVectors)
{
  assert(m1.size() == m2.size()
	 && m1.type() == m2.type()
	 && m1.type() == CV_8UC1);
  assert(nbLevels > 0);

  motionVectors.resize(nbLevels);
  levels1.resize(nbLevels);
  levels2.resize(nbLevels);
  levels1[0] = m1;
  levels2[0] = m2;
  for (int i=1; i<nbLevels; ++i) { //start from 1
    cv::pyrDown(levels1[i-1], levels1[i], cv::Size(levels1[i-1].cols/2, levels1[i-1].rows/2));
    cv::pyrDown(levels2[i-1], levels2[i], cv::Size(levels2[i-1].cols/2, levels2[i-1].rows/2));
  }

  for (int i=nbLevels-1; i>=0; i--) {

    computeOpticalFlowHornSchunckMonoAux(levels1[i], levels2[i], lambda, criteria, motionVectors[i]); 

    if (i > 0) {
      up(motionVectors[i], levels2[i-1].size(), motionVectors[i-1]); 
    }
  }
  
}
