/*

  //OpenCV documentation is available here: http://docs.opencv.org/2.4.9/

  //Global Motion Estimation

*/

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <queue>
#include <sstream>


#include <opencv2/highgui/highgui.hpp> //VideoCapture, imshow, imwrite, ...
#include <opencv2/imgproc/imgproc.hpp> //cvtColor

#include "utils.hpp"
#include "motionEstimation.hpp"
#include "GME.hpp"
#include <opencv2/video/video.hpp> //calcOpticalFlowFarneback


#define DISPLAY 1
#define SAVE_FRAME 0

int
main(int argc, char **argv)
{
  if(argc != 4) {
    std::cerr << "Usage: " << argv[0] << " video-filename distance-between-two-frames-for-prediction nbLevels" << std::endl;
    return EXIT_FAILURE;
  }

  const char *videoFilename = argv[1];

  const int interFramesDistance = atoi(argv[2]);
  if (interFramesDistance <= 0) {
    std::cerr<<"Error: distance-between-two-frames-for-prediction must be a strictly positive integer"<<std::endl;
    return EXIT_FAILURE;
  }

  const int nbLevels = atoi(argv[3]);
  if (nbLevels <= 0 || nbLevels>5) {
    std::cerr<<"Error: nbLevels must be a strictly positive integer"<<std::endl;
    return EXIT_FAILURE;
  }

  cv::VideoCapture cap;
  cap.open(videoFilename);
  if ( ! cap.isOpened()) {
    std::cerr << "Error : Unable to open video file " << argv[1] << std::endl;
    return -1;
  }

  unsigned long frameNumber = 0;

  const size_t deltaT = interFramesDistance;
  std::queue<cv::Mat> previousFrames;  

  for ( ; ; ) {

    cv::Mat frameBGR;
    cap >> frameBGR;
    if (frameBGR.empty()) {
      break;
    }

    //save frame
#if SAVE_FRAME
    std::stringstream ss;
    ss<<"frame_"<<std::setfill('0')<<std::setw(6)<<frameNumber<<".png";
    cv::imwrite(ss.str(), frameBGR);
#endif //SAVE_FRAME
    
    //convert from BGR to Y
    cv::Mat frameY;
    cv::cvtColor(frameBGR, frameY, CV_BGR2GRAY);

    if (previousFrames.size() >= deltaT) {
      cv::Mat prevY = previousFrames.front();
      previousFrames.pop();
  
      //TODO: compute the motion vectors with one of the motion estimation methods (H&S, Farnback or MIT)
      cv::Mat motionVectors;
      //computeMotionEstimation(prevY, frameY, motionVectors);
      const int nbIter = 20; //20;
      const int levels = nbLevels;
      const int winSize = 20; //15;
      const int poly_n = 5;
      const double poly_sigma = 1.2; //1.1;
      const double pyr_scale = 0.5;
      const int flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
      //TODO: call calcOpticalFlowFarneback()
      calcOpticalFlowFarneback(frameY, prevY, motionVectors, pyr_scale, levels, winSize, nbIter, poly_n, poly_sigma, flags);
      
      cv::Mat motionVectorsGlobal;
      computeGME(motionVectors, motionVectorsGlobal);

      cv::Mat motionError;
      computeGlobalMotionError(motionVectors, motionVectorsGlobal, motionError);
      
      cv::Mat YC;
      //TODO: compute compensated image YC 
      computeCompensatedImageF(motionVectors, prevY, YC);

      cv::Mat YC2;
      //TODO: compute compensated image YC2 with motionVectorsGlobal
      computeCompensatedImageF(motionVectorsGlobal, prevY, YC2);
	
#if DISPLAY
      cv::Mat imgMV = frameY.clone();
      drawMVf(imgMV, motionVectors);
      cv::imshow("MVs", imgMV);
      cv::Mat imgMV2 = frameY.clone();
      drawMVf(imgMV2, motionVectorsGlobal);
      cv::imshow("GMVs", imgMV2);
      cv::waitKey(10);
#endif //DISPLAY
	
      cv::Mat imErr0;
      computeErrorImage(frameY, prevY, imErr0);
      
      cv::Mat imErr;
      computeErrorImage(frameY, YC, imErr);
      
      const double MSE0 = computeMSE(frameY, prevY);
      const double MSE = computeMSE(frameY, YC);
      const double PSNR = computePSNR(MSE);
      const double ENT = computeEntropy(frameY);
      const double ENTe = computeEntropy(imErr);
      const double MSE2 = computeMSE(frameY, YC2);
     
      std::cout<<frameNumber<<" "<<MSE<<" "<<PSNR<<" "<<ENT<<" "<<ENTe<<" "<<MSE0<<" "<<MSE2<<std::endl;
    }

    previousFrames.push(frameY);    
      
    ++frameNumber;
  }

  return EXIT_SUCCESS;
}
