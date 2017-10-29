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

  std::ofstream stats("stats.txt");
  std::ofstream statsEntropy("statsEntropy.txt");

  std::string name[nbLevels];
  for (int i = nbLevels-1; i >= 0; i--)
    name[i] =  "image" + std::to_string(i);

  for ( ; ; ) {

    cv::Mat frameBGR;
    cap >> frameBGR;

    cv::Mat frameError(frameBGR.rows, frameBGR.cols, CV_8UC1);

    if (frameBGR.empty()) {
      break;
    }

    //save frame
    std::stringstream ss;
    ss<<"Resultats/frame_"<<std::setfill('0')<<std::setw(6)<<frameNumber<<".png";
    cv::imwrite(ss.str(), frameBGR);

    //convert from BGR to Y
    cv::Mat frameY;
    cv::cvtColor(frameBGR, frameY, CV_BGR2GRAY);

    if (previousFrames.size() >= deltaT) {
      cv::Mat prevY = previousFrames.front();
      previousFrames.pop();

      double MSE, PSNR, ENT, ENTe;

      if (nbLevels == 1) {

      	cv::Mat motionVectors;
      	blockMatchingMono(frameY, prevY, blockSize, windowSize, motionVectors);
      	cv::Mat YC;
      	computeCompensatedImage(motionVectors, prevY, YC);

      	//TODO: compute measures (& display images) ...

        cv::Mat tmp = frameY.clone();
        drawMVi(tmp, motionVectors);
        imshow("tmp", YC);
        cv::waitKey(10);

        MSE = computeMSE(previousFrames.front(), YC);
        PSNR = computePSNR(MSE);
        ENT = computeEntropy(YC);
        computeErrorImage(previousFrames.front(), YC, frameError);
        ENTe = computeEntropy(frameError);

        stats << frameNumber << " " << MSE << " " << PSNR << std::endl;
        statsEntropy << frameNumber << " " << ENT << " " << ENTe << std::endl;

      	std::cout<<frameNumber<<" "<<MSE<<" "<<PSNR<<" "<<ENT<<" "<<ENTe<<"\n";
      }
      else {

	std::vector<cv::Mat> levelsY;
	std::vector<cv::Mat> levelsPrevY;
	std::vector<cv::Mat> motionVectorsP;
	blockMatchingMulti(frameY, prevY, blockSize, windowSize, nbLevels, levelsY, levelsPrevY, motionVectorsP);

	std::cout<<frameNumber;
  double cpt(0);
	for (int i=nbLevels-1; i>=0; --i) {

	  //TODO : compute measures  (& display images) ...

    cv::Mat YC(levelsPrevY[i].rows, levelsPrevY[i].cols, CV_8UC1);
    computeCompensatedImage(motionVectorsP[i], levelsY[i], YC);
    cv::Mat tmp = YC.clone();
    //drawMVi(tmp, motionVectorsP[i]);
    //imshow(name[i], tmp);
    //cv::waitKey(10);

    MSE = computeMSE(levelsPrevY[i], YC);
    PSNR = computePSNR(MSE);
    ENT = computeEntropy(YC);
    computeErrorImage(levelsPrevY[i], YC, frameError);
    ENTe = computeEntropy(frameError);

    if (cpt == 2) {
      stats << frameNumber << " " << MSE << std::endl;
      cpt=0;
    }
    else {
      stats << frameNumber << " " << MSE;
      cpt++;
    }
	  std::cout<< "  " << i << " || MSE : "<<MSE<<" || PSNR : "<<PSNR<<" || ENT : "<<ENT<<" || ENTe : "<<ENTe << std::endl;
	}
	std::cout<<"\n";
      }


    }

    previousFrames.push(frameY);

    ++frameNumber;
  }

  return EXIT_SUCCESS;
}
