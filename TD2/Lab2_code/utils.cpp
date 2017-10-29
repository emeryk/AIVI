#include "utils.hpp"

#include <opencv2/imgproc/imgproc.hpp>

void
computeErrorImage(const cv::Mat &im, const cv::Mat &imC, cv::Mat &imErr)
{
  assert(im.size() == imC.size()
	 && im.type() == imC.type()
	 && imC.type() == CV_8UC1);

  //We compute a displyable error image
  //We keep pixel values in [0; 255]

  imErr.create(imC.rows, imC.cols, CV_8UC1); //nothing done if matrix already has the desired size
  int rows = im.rows;
  int cols = im.cols;
  if (im.isContinuous() && imC.isContinuous() && imErr.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for (int y = 0; y < rows; ++y) {
    const unsigned char *p1 = im.ptr<unsigned char>(y);
    const unsigned char *p2 = imC.ptr<unsigned char>(y);
    unsigned char *pe = imErr.ptr<unsigned char>(y);
    for(int x = 0; x < cols; ++x) {
      pe[x] = cv::saturate_cast<unsigned char>(p1[x]-p2[x] + 128);
    }
  }

}


#define SQUARE(x) ((x) * (x))

double computeMSE(const cv::Mat &im, const cv::Mat &imC)
{
  assert(im.size() == imC.size()
	 && im.type() == imC.type()
	 && imC.type() == CV_8UC1);

  long sum = 0;

  int rows = im.rows;
  int cols = im.cols;
  if (im.isContinuous() && imC.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for (int y = 0; y < rows; ++y) {
    const unsigned char *p1 = im.ptr<unsigned char>(y);
    const unsigned char *p2 = imC.ptr<unsigned char>(y);
    for(int x = 0; x < cols; ++x) {
      sum += SQUARE((long)(p1[x]) - (long)(p2[x])); //Warning: do not compute with unsigned chars
    }
  }

  const double MSE = sum / (double)(im.cols * im.rows);
  return MSE;
}

double computePSNR(const cv::Mat &im, const cv::Mat &imC)
{
  return computePSNR(computeMSE(im, imC));
}

double computePSNR(double MSE)
{
  const double PSNR = 10 * std::log10(SQUARE(255) / MSE);
  return PSNR;
}

double computeEntropy(const cv::Mat &im)
{
  assert(im.type() == CV_8UC1);

  const int size = 256;
  long histogram[size] = {0};

  int rows = im.rows;
  int cols = im.cols;
  if (im.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for (int y = 0; y < rows; ++y) {
    const unsigned char *p1 = im.ptr<unsigned char>(y);
    for(int x = 0; x < cols; ++x) {
      histogram[p1[x]] += 1;
    }
  }
  const double norm = rows*cols;

  double entropy = 0.0;
  for (int i=0; i<size; ++i) {
    if (histogram[i] > 0) {
      const double pi = (double)(histogram[i]) / norm;

	entropy -= pi * log2(pi);
      }
  }

  return entropy;
}


void
drawMV(cv::Mat &img, const cv::Point &pt1, const cv::Point &pt2,
       const cv::Scalar &color, const int thickness)
{
  cv::Point p1 = pt1;
  cv::Point p2 = pt2;

#if 1
  const double angle = atan2( (double) p1.y - p2.y, (double) p1.x - p2.x );
  //const double hypotenuse = sqrt( SQUARE(p1.y - p2.y) + SQUARE(p1.x - p2.x) );

  cv::line(img, p1, p2, color, thickness, CV_AA, 0);

  const double scaling_factor = 2;
  p1.x = (int) (p2.x + scaling_factor * cos(angle + M_PI / 4));
  p1.y = (int) (p2.y + scaling_factor * sin(angle + M_PI / 4));
  cv::line(img, p1, p2, color, thickness, CV_AA, 0);
  p1.x = (int) (p2.x + scaling_factor * cos(angle - M_PI / 4));
  p1.y = (int) (p2.y + scaling_factor * sin(angle - M_PI / 4));
  cv::line(img, p1, p2, color, thickness, CV_AA, 0);
#else

  cv::line(img, p1, p2, color, thickness, CV_AA, 0);
  //cv::circle(img, p1, 2, color, -1);
  cv::circle(img, p2, 2, color, -1);

#endif
}

template <typename T>
void
drawMV(cv::Mat &img,
       const cv::Mat &motionVectors,
       int step,
       const cv::Scalar &color,
       const int thickness)
{
  const int blockSize = img.rows / motionVectors.rows;

  const int halfBlockSize = blockSize / 2 ;

  const int nbY = img.rows / blockSize;
  const int nbX = img.cols / blockSize;

  assert(nbX == motionVectors.cols);
  assert(nbY == motionVectors.rows);

  for(int y = 0; y < nbY; y+=step) {

    const T *p = motionVectors.ptr<T>(y);

    for(int x = 0; x < nbX; x+=step) {

      const float mv_x = (p[x])[0];
      const float mv_y = (p[x])[1];

      size_t x1 = x * blockSize + halfBlockSize;
      size_t y1 = y * blockSize + halfBlockSize;

      size_t x2 = x1 + mv_x;
      size_t y2 = y1 + mv_y;

      const cv::Point p1 = cv::Point2i(x1, y1);
      const cv::Point p2 = cv::Point2i(x2, y2);
      drawMV(img, p2, p1, color, thickness);
    }
  }

}


void
drawMVi(cv::Mat &img,
	const cv::Mat &motionVectors,
	int step,
	const cv::Scalar &color,
	const int thickness)
{
  assert(motionVectors.type() == CV_32SC2);
  drawMV<cv::Vec2i>(img, motionVectors, step, color, thickness);
}


void
computeCompensatedImage(const cv::Mat &motionVectors,
			const cv::Mat &prev,
			cv::Mat &compensated)
{

  assert(motionVectors.type() == CV_32SC2);
  assert(prev.type() == CV_8UC1);

  const int blockSize = prev.cols/motionVectors.cols;
  //TODO: fill compensated

  const int blockX = motionVectors.cols;
  const int blockY = motionVectors.rows;

  compensated = prev.clone();

  for (int i(0); i < blockY; i++) {
    for (int j(0); j < blockX; j++) {

      cv::Vec2i vector = motionVectors.at<cv::Vec2i>(i,j);

      cv::Mat block(prev, cv::Rect(j * blockSize + vector(0), i * blockSize + vector(1), blockSize, blockSize));

      block.copyTo(compensated(cv::Rect(j * blockSize, i * blockSize, blockSize, blockSize)));
    }
  }

  //You can use rowRange/colRange to get blocks of images
  // and copyTo to copy these blocks

}
