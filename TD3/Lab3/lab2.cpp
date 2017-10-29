/*

  //OpenCV documentation is available here: http://docs.opencv.org/2.4.9/

  //Block matching in mono and multi-resolution

*/

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <queue>
#include <sstream>
#include <fstream>


#include <opencv2/highgui/highgui.hpp> //VideoCapture, imshow, imwrite, ...
#include <opencv2/imgproc/imgproc.hpp> //cvtColor

#include "utils.hpp"
#include "blockmatching.hpp"

#define DISPLAY 1

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
  if (nbLevels <= 0 || nbLevels>4) {
    std::cerr<<"Error: nbLevels must be a strictly positive integer"<<std::endl;
    return EXIT_FAILURE;
  }

  const int blockSize = 8;
  const int windowSize = 32;
  //TODO: it would be better to pass these values as parameters

  cv::VideoCapture cap;
  cap.open(videoFilename);
  if ( ! cap.isOpened()) {
    std::cerr << "Error : Unable to open video file " << argv[1] << std::endl;
    return -1;
  }

  unsigned long frameNumber = 0;

  const size_t deltaT = interFramesDistance;
  std::queue<cv::Mat> previousFrames;  

	std::ofstream res("result.txt");

  for ( ; ; ) {

    cv::Mat frameBGR;
    cap >> frameBGR;
    if (frameBGR.empty()) {
      break;
    }

    //save frame
    /*
    std::stringstream ss;
    ss<<"frame_"<<std::setfill('0')<<std::setw(6)<<frameNumber<<".png";
    cv::imwrite(ss.str(), frameBGR);
    */
    
    //convert from BGR to Y
    cv::Mat frameY;
    cv::cvtColor(frameBGR, frameY, CV_BGR2GRAY);

    if (previousFrames.size() >= deltaT) {
      cv::Mat prevY = previousFrames.front();
      previousFrames.pop();
/*
      if (nbLevels == 1) {

	cv::Mat motionVectors;
	blockMatchingMono(frameY, prevY, blockSize, windowSize, motionVectors);
	cv::Mat YC;
	computeCompensatedImage(motionVectors, prevY, YC);
	
#if DISPLAY
	cv::Mat imgMV = frameY.clone();
	drawMVi(imgMV, motionVectors);
	
	cv::imshow("MVs", imgMV);
	cv::waitKey(10);
#endif //DISPLAY
	
	//TODO: FD ???????
	
	cv::Mat imErr0;
	computeErrorImage(frameY, prevY, imErr0);
	
	cv::Mat imErr;
	computeErrorImage(frameY, YC, imErr);
	
	const double MSE = computeMSE(frameY, YC);
	const double PSNR = computePSNR(MSE);
	const double ENT = computeEntropy(frameY);
	const double ENTe = computeEntropy(imErr);
	
	std::cout<<frameNumber<<" "<<MSE<<" "<<PSNR<<" "<<ENT<<" "<<ENTe<<std::endl;
      }
      else {
	//*/
	std::vector<cv::Mat> levelsY;
	std::vector<cv::Mat> levelsPrevY;
	std::vector<cv::Mat> motionVectorsP;
	blockMatchingMulti(frameY, prevY, blockSize, windowSize, nbLevels, levelsY, levelsPrevY, motionVectorsP);

	//TODO : compute measures ... 

	std::cout<<frameNumber;
	for (int i=nbLevels-1; i>=0; --i) {
	  const cv::Mat &motionVectors = motionVectorsP[i];
	  const cv::Mat &prevY = levelsPrevY[i];
	  const cv::Mat &frameY = levelsY[i];
	  
	  cv::Mat YC;
	  computeCompensatedImage(motionVectors, prevY, YC);

#if DISPLAY
	  cv::Mat imgMV = frameY.clone();
	  drawMVi(imgMV, motionVectors);
	  
	  cv::imshow(std::string("MVs")+std::to_string(i), imgMV);
	  cv::waitKey(10);
#endif //DISPLAY
	  
	  //TODO: FD ???????
	  
	  cv::Mat imErr0;
	  computeErrorImage(frameY, prevY, imErr0);
	  
	  cv::Mat imErr;
	  computeErrorImage(frameY, YC, imErr);
	  
	  const double MSE = computeMSE(frameY, YC);
	  const double PSNR = computePSNR(MSE);
	  const double ENT = computeEntropy(frameY);
	  const double ENTe = computeEntropy(imErr);
	  
	  
	  std::cout<<" "<<MSE<<" ";//<<PSNR<<" "<<ENT<<" "<<ENTe;
	  res << frameNumber <<" "<<MSE<<" ";
	}
	res << std::endl;
	std::cout<<"\n";
     // }
	
      
    }

    previousFrames.push(frameY);    
      
    ++frameNumber;
  }

  return EXIT_SUCCESS;
}
