#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core/core.hpp>

//compute error image
//im & imC are input images, imErr is the output image
extern void computeErrorImage(const cv::Mat &im, const cv::Mat &imC, cv::Mat &imErr);

//compute MSE between two images
extern double computeMSE(const cv::Mat &im, const cv::Mat &imC);

//compute PSNR between two images
extern double computePSNR(const cv::Mat &im, const cv::Mat &imC);

//compute PSNR from already computed MSE
extern double computePSNR(double MSE);

//compute entropy of an image
extern double computeEntropy(const cv::Mat &im);

//compute motion compensated image
// motionVectors is of type CV_32SC2
extern void computeCompensatedImage(const cv::Mat &motionVectors,
				    const cv::Mat &prev,
				    cv::Mat &compensated);

//compute motion compensated image
// motionVectors is of type CV_32FC2
extern void computeCompensatedImageF0(const cv::Mat &motionVectors,
				      const cv::Mat &prev,
				      cv::Mat &compensated);

//compute motion compensated image
// motionVectors is of type CV_32FC2
extern void computeCompensatedImageF(const cv::Mat &motionVectors,
				     const cv::Mat &prev,
				     cv::Mat &compensated);



//Draw motionVectors on image img.
//img is both input and output.
//motionVectors are of type CV_32SC2.
extern void drawMVi(cv::Mat &img, 
		    const cv::Mat &motionVectors, 
		    int step = 4,
		    const cv::Scalar &color = cv::Scalar(255, 255, 255), 
		    const int thickness = 1); 

//Draw motionVectors on image img.
//img is both input and output.
//motionVectors are of type CV_32FC2.
extern void drawMVf(cv::Mat &img, 
		    const cv::Mat &motionVectors, 
		    int step = 4,
		    const cv::Scalar &color = cv::Scalar(255, 255, 255), 
		    const int thickness = 1); 


#endif /* ! UTILS_HPP */
