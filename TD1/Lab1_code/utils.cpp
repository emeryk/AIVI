#include "utils.hpp"
#include "math.h"
#include <cassert>

void
computeErrorImage(const cv::Mat &im, const cv::Mat &imC, cv::Mat &imErr)
{
  //TODO...
  assert(im.size() == imC.size());
  assert(im.type() == imC.type());
  assert(im.type() == imErr.type());

  int err = 0;

  for (int i(0); i < im.rows; i++)
  {
    const uchar *r1 = im.ptr<uchar>(i);
    const uchar *r2 = imC.ptr<uchar>(i);
    for (int j(0); j < im.cols; j++)
    {
      err = (int)r1[j] - (int)r2[j] + 128;
      if(err < 0) err = 0;
      if(err > 255) err = 255;
      imErr.data[i * im.cols + j] = (const uchar)err;
    }
  }
}

double computeMSEOld(const cv::Mat &im, const cv::Mat &imC)
{
  int M = im.size().width;
  int N = im.size().height;

  double MSE = 0.0f;

  cv::MatConstIterator_<uchar> iterator = im.begin<uchar>();
  cv::MatConstIterator_<uchar> iterator2 = imC.begin<uchar>();

  for(; iterator != im.end<uchar>(); iterator++, iterator2++)
  {
    double res = (*iterator - *iterator2);
    MSE += res * res;
  }

  return MSE / (M*N);
}

double computeMSE(const cv::Mat &im, const cv::Mat &imC)
{
  assert(im.size() == imC.size());
  assert(im.type() == imC.type());
  //assert(im.type() == CV_8UC1);
  double mse = 0;
  for (int i = 0; i < im.rows; i++)
  {
    //const uchar *r1 = im.ptr<uchar>(i);
    //const uchar *r2 = imC.ptr<uchar>(i);
    for (int j = 0; j < im.cols; j++)
    {
      mse += pow(((int)im.at<uchar>(i,j) - (int)imC.at<uchar>(i,j)), 2);
      //mse += pow((int)r1[j] - (int)r2[j], 2);
    }
  }
  return mse / (im.cols * im.rows);
}

double computePSNR(const cv::Mat &im, const cv::Mat &imC)
{
  double mse = computeMSE(im, imC);
  return (10 * log10(pow(255, 2) / mse));
}

double computePSNR(double MSE)
{
  return (10 * log10(pow(255, 2) / MSE));
}

double computeEntropy(const cv::Mat &im)
{
  double histo[256];
  for (int i(0); i < 256; i++)
    histo[i] = 0;
  double entropy = 0;

  for (int i(0); i < im.rows; i++)
  {
    const uchar *r1 = im.ptr<uchar>(i);
    for (int j(0); j < im.cols; j++)
    {
        histo[(int)r1[j]]++;
    }
  }

  double size = im.rows * im.cols;

  for (int i(0); i < 256; i++)
  {
    histo[i] /= size;
    if(histo[i] == 0)
      entropy += 0;
    else
      entropy += histo[i] * log2(histo[i]);
  }

  return (entropy * -1);
}
