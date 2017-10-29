/*

  //OpenCV documentation is available here: http://docs.opencv.org/2.4.9/

  //1.1a - Display & save individual frames of the video as RGB
  // You will need to use classes & methods from the highgui module:
  // VideoCapture, imwrite, imshow, ...
  // You have an example of use of VideoCapture here:
  // http://docs.opencv.org/2.4.9/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture

  //1.1b - Display & save individual frames of the video as Y from YCrCb
  // You will need to use:
  //  cvtColor from the imgproc module
  //  split from the core module

  //1.2 - Compute the measures: MSE, PSNR, Entropy, Error Image
  // You will need to code these measures in utils.hpp/utils.cpp

  //1.3 - Trace the curves with gnuplot
  // You have examples of gnuplot commands are available in files example*.gplot

*/

#include <cstdlib>
#include <iostream>

#include <queue>
#include <fstream>

#include "utils.hpp"

#include <opencv2/highgui/highgui.hpp> //VideoCapture, imshow, imwrite, ...
#include <opencv2/imgproc/imgproc.hpp> //cvtColor

int
main(int argc, char **argv)
{
  if(argc != 3) {
    std::cerr << "Usage: " << argv[0] << " video-filename distance-between-two-frames-for-prediction" << std::endl;
    return EXIT_FAILURE;
  }

  const char *videoFilename = argv[1];

  const int interFramesDistance = atoi(argv[2]);
  if (interFramesDistance <= 0) {
    std::cerr<<"Error: distance-between-two-frames-for-prediction must be a strictly positive integer"<<std::endl;
    return EXIT_FAILURE;
  }

  cv::VideoCapture cap;
  cap.open(videoFilename);
  if ( ! cap.isOpened()) {
    std::cerr << "Error : Unable to open video file " << argv[1] << std::endl;
    return -1;
  }

  unsigned long frameNumber = 0;

  cv::namedWindow("Video", 1);
  std::queue<cv::Mat> workingQueue;

  std::ofstream stats("stats.txt");
  std::ofstream statsEntropy("statsEntropy.txt");


  for ( ; ; ) {

    cv::Mat frameBGR;
    cv::Mat workingMat;
    cap >> frameBGR;

    cv::Mat tmp[3];
    cv::Mat frameError(frameBGR.rows, frameBGR.cols, CV_8UC1);

    double mse, psnr, entropy, entropyErr;

    if (frameBGR.empty()) {
      break;
    }

    cv::cvtColor(frameBGR, workingMat, CV_BGR2GRAY); //CV_BGR2YUV CV_BGR2HSV CV_BGR2YCrCb
    imshow("Video", workingMat);

    if(frameNumber < interFramesDistance)
      workingQueue.push(workingMat);

    else{
      mse = computeMSE(workingQueue.front(), workingMat);
      psnr = computePSNR(mse);
      entropy = computeEntropy(workingMat);
      computeErrorImage(workingQueue.front(), workingMat, frameError);
      entropyErr = computeEntropy(frameError);

      workingQueue.pop();
      workingQueue.push(workingMat);

      //stats << "Frame Number : " <<frameNumber << " | MSE : " << mse << " | PSNR : " << psnr << std::endl;
      //statsEntropy << "Frame Number : " << frameNumber << " | Entropy : " << entropy << " | Entropy Error : " << entropyErr << std::endl;
      stats << frameNumber << " " << mse << " " << psnr << std::endl;
      statsEntropy << frameNumber << " " << entropy << " " << entropyErr << std::endl;


      std::string filename = "Resultats/frame" + std::to_string(frameNumber) + ".png";
      cv::split(frameError, tmp);
      cv::imwrite(filename, tmp[0]);
    }

    if(cv::waitKey(interFramesDistance) >= 0) break;

    ++frameNumber;
  }

  /*for (int i(0); i < frameNumber-1; i++)
  {
    cv::Mat currentFrame = cv::imread("Resultats/frame" + std::to_string(i) + ".png");
    cv::Mat nextFrame = cv::imread("Resultats/frame" + std::to_string(i+1) + ".png");
    double mse = computeMSE(currentFrame, nextFrame);
    //double mse2 = computeMSEOld(currentFrame, nextFrame);
    double psnr = computePSNR(mse);
    //double psnr2 = computePSNR(currentFrame, nextFrame);
    double ent = computeEntropy(currentFrame);
    printf("MSE : %f\nPSNR : %f\nENT : %f\n", mse, psnr, ent);
  }*/

  stats.close();
  statsEntropy.close();

  return EXIT_SUCCESS;
}
