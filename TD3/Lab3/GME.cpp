#include "GME.hpp"

#include <opencv2/calib3d/calib3d.hpp> //findHomography

#include <iostream>

void
computeGME(const cv::Mat &motionVectors,
	   cv::Mat &motionVectorsGlobal)
{
  //TODO: compute global motion estimated vectors
  // use cv::findHomeography and cv::perspectiveTransform

  //See http://docs.opencv.org/2.4.9/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findhomography
  // and https://docs.opencv.org/2.4.9/modules/core/doc/operations_on_arrays.html?highlight=perspectivetransform#cv2.perspectiveTransform
  //According to the documentation, cv::findHomography & cv::perspectiveTransform take InputArrays & OutputArrays
  //See here http://docs.opencv.org/2.4.13/modules/core/doc/intro.html#inputarray-and-outputarray
  //They are just proxies around a cv::Mat or a std::vector
  //For example, for cv::findHomography(), for 'src',  we can pass a cv::Mat of type CV_32FC2 or a std::vector<cv::Point2f>. It will work the same.
  //
  //**WARNING**
  //If you pass cv::Mats to cv::findHomography, they have to have 1 column or 1 row !
  //Otherwise, there is an assert that will fail in OpenCV implementation.
  //You can use cv::reshape to just change the shape of you matrix :
  // pass from a matrix of size rows x cols to a matrix of size 1 x rows*cols.
  //Here you would call: reshape(0, 1);

	std::vector<cv::Point2f> srcPts;
	std::vector<cv::Point2f> dstPts;

	for (int i(0); i < motionVectors.rows; i++)
	{
		for (int j(0); j < motionVectors.cols; j++)
		{
			srcPts.push_back(cv::Point2f(i, j));
			dstPts.push_back(i + motionVectors.at<cv::Vec2f>(i,j)[0], j + motionVectors.at<cv::Vec2f>(i,j)[1]);
		}
	}

	cv::Mat homography = cv::findHomography(srcPts, dstPts, CV_RANSAC);

	motionVectorsGlobal.create(motionVectors.rows, motionVectors.cols, CV_32FC2);

	cv::perspectiveTransform(srcPts, dstPts, homography);

	for (int i(0); i < motionVectors.rows; i++)
	{
		for (int j(0); j < motionVectors.cols; j++)
		{
			motionVectorsGlobal.at<cv::Vec2f>(i,j) = dstPts.at(i * motionVectors.cols + j) - srcPts.at(i * motionVectors.cols + j);
		}
	}

  assert(motionVectorsGlobal.type() == CV_32FC2);

}

void
computeGlobalMotionError(const cv::Mat &motionVectors,
			 const cv::Mat &motionVectorsGlobal,
			 cv::Mat &motionError)
{
  assert(motionVectors.type() == CV_32FC2);
  assert(motionVectorsGlobal.type() == CV_32FC2);

  motionError.create(motionVectors.rows, motionVectors.cols, CV_32F);

  //TODO: compute the error between actual motion vectors and estimated global motion vectors

	float maxErr = 0.0f;

	for (int i(0); i < motionVectors.rows; i++)
	{
		for (int j(0); j < motionVectors.cols; j++)
		{
			float x = motionVectors.at<cv::Vec2f>(i,j)[0] - motionVectorsGlobal.at<cv::Vec2f>(i,j)[0];
			float y = motionVectors.at<cv::Vec2f>(i,j)[1] - motionVectorsGlobal.at<cv::Vec2f>(i,j)[1];

			motionError.at<float>(i,j) = sqrt(x * x + y * y);

			if (motionError.at<float>(i,j) > maxErr)
				maxErr = motionError.at<float>(i,j);
		}
	}

  assert(motionError.type() == CV_32F);
}
