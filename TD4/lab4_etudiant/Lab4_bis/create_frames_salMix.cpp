#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
void
usage(const char *cmd)
{
	std::cerr << "Usage: " << cmd << " file-list-frames file-saliency-frames out_folder temp_window_size_for_sal_smooting(optional)" << std::endl;
	exit(EXIT_FAILURE);
}

enum frame_position{
	left,
	bottom_right,
	none
};

int getNbLinesFile(char * File);
cv::Mat getSmoothedSaliency(int lineCurrentSaliency, int temp_window_size, int nb_frames, char *saliencyListFile);

int
main(int argc, char **argv)
{
	if(argc < 4 || argc > 5)
		usage(argv[0]);

	int temp_window_size = 0;
	if (argc ==5){
		temp_window_size = (atoi(argv[4])%100 == 0)?(atoi(argv[4])+1):atoi(argv[4]);
	}

	std::string lineFrames;
	std::string lineSaliencies;
	std::ifstream listFramesFile(argv[1]);
	std::ifstream listSalienciesFile(argv[2]);

	cv::Mat frame;
	cv::Mat saliency;
	cv::Mat mix;
	cv::Mat output_frame;

	int output_width;
	int nbPixelsForText = 0;

	float alpha = 0.6; // between [0,1] = prop of opacity given to the original frame

	frame_position fp = none;
	float miniatureBy = 4;

	std::string extension = ".png";
	std::string title;

	int count=1;
	int nb_frames=getNbLinesFile(argv[1]);
	int nb_sal=getNbLinesFile(argv[2]);
	int nb_iterations = (nb_frames >= nb_sal)?nb_sal:nb_frames;
	int step_progress = 100;
	std::cout << "nb_frames=" << nb_frames  << "nb_sal="  << nb_sal << std::endl;

	std::cout << count << "/" << nb_iterations << std::endl;

	if (listFramesFile.is_open() && listSalienciesFile.is_open())
	{
		while ( listFramesFile.good() && listSalienciesFile.good())
		{
			// *** read the pictures ***
			listFramesFile >> lineFrames;
			listSalienciesFile >> lineSaliencies;

			frame = cv::imread(lineFrames, CV_LOAD_IMAGE_COLOR);   // Read the file
			if(! frame.data )                              // Check for invalid input
			{
				std::cout <<  "Could not open or find the image:" <<  lineFrames << std::endl ;
				exit(EXIT_FAILURE);
			}

			if (temp_window_size>0){
				saliency = getSmoothedSaliency(count, temp_window_size, nb_frames, argv[2]);
			}
			else{
				saliency = cv::imread(lineSaliencies, CV_LOAD_IMAGE_GRAYSCALE );
			}


			if(! saliency.data )                              // Check for invalid input
			{
				std::cout <<  "Could not open or find the image:" <<  lineSaliencies << std::endl ;
				exit(EXIT_FAILURE);
			}

			// *** create saliency-on-frame picture ***
			cv::applyColorMap(saliency, saliency, cv::COLORMAP_JET);
			mix= alpha*frame + (1-alpha)*saliency;

			// *** create the output frame = (original frame + mixed frame)
			if (fp==left){
				output_width = frame.cols + mix.cols + nbPixelsForText;
				output_frame = cv::Mat(frame.rows, output_width, CV_8UC3, cv::Scalar(0));

				cv::Mat aux_frame = output_frame.colRange(nbPixelsForText, nbPixelsForText+frame.cols).rowRange(0, frame.rows);
				frame.copyTo(aux_frame);
				cv::Mat aux_mix = output_frame.colRange(nbPixelsForText+frame.cols, nbPixelsForText+frame.cols+mix.cols).rowRange(0, frame.rows);
				mix.copyTo(aux_mix);
			}
			else if (fp==bottom_right){
				output_width = mix.cols + nbPixelsForText;
				output_frame = cv::Mat(mix.rows, output_width, CV_8UC3, cv::Scalar(0));

				cv::Mat frame_mini;
				cv::resize(frame, frame_mini, cv::Size(), 1/miniatureBy, 1/miniatureBy);

				cv::Mat aux_mix = output_frame.colRange(nbPixelsForText, nbPixelsForText+mix.cols).rowRange(0, mix.rows);
				mix.copyTo(aux_mix);
				cv::Mat aux_frame = output_frame.colRange(nbPixelsForText + frame.cols*(1-1/miniatureBy), nbPixelsForText+frame.cols).rowRange(frame.rows*(1-1/miniatureBy), frame.rows);
				frame_mini.copyTo(aux_frame);
			}
			else if (fp==none){
				output_width = mix.cols + nbPixelsForText;
				output_frame = cv::Mat(mix.rows, output_width, CV_8UC3, cv::Scalar(0));

				cv::Mat aux_mix = output_frame.colRange(nbPixelsForText, nbPixelsForText+mix.cols).rowRange(0, mix.rows);
				mix.copyTo(aux_mix);
			}

			std::ostringstream count_str;
			count_str << "/" << std::setw(8) << std::setfill('0') << count;
			title=argv[3]+count_str.str()+extension;

			cv::imwrite(title, output_frame);

			count++;

			/*Display progress*/
			if (count%step_progress == 0){
				std::cout << count << "/" << nb_iterations << std::endl;
			}

		}
		listFramesFile.close();
	}
	else{
		std::cerr << "Cannot open file: " << argv[1] << std::endl;
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}

cv::Mat getSmoothedSaliency(int lineCurrentSaliency, int temp_window_size, int nb_frames, char *saliencyListFile)
{
	cv::Mat smoothedSal;

	int line_start = (lineCurrentSaliency-(temp_window_size-1)/2 < 1)?1:(lineCurrentSaliency-(temp_window_size-1)/2);
	int line_end = (lineCurrentSaliency+(temp_window_size-1)/2 > nb_frames)?nb_frames:(lineCurrentSaliency+(temp_window_size-1)/2);

	std::ifstream salFileStream(saliencyListFile);
	std::string currentSal;

	int currIdx=1;
	int first_pass=1;

	if (salFileStream.is_open())
	{
		while ( salFileStream.good() && currIdx <= line_end )
		{
			salFileStream >> currentSal;

			if (currIdx >= line_start)
			{
				if (first_pass == 1){
					smoothedSal = cv::imread(currentSal, CV_LOAD_IMAGE_GRAYSCALE);
					smoothedSal.convertTo(smoothedSal, CV_32F);
					first_pass = 0;
				}
				else{
					cv::Mat imAux = cv::imread(currentSal, CV_LOAD_IMAGE_GRAYSCALE);
					imAux.convertTo(imAux, CV_32F);
					cv::add(smoothedSal, imAux, smoothedSal);
				}
			}
			currIdx++;
		}
		salFileStream.close();
	}
	else{
		std::cerr << "Cannot open file: " << saliencyListFile << std::endl;
		exit(EXIT_FAILURE);
	}

	smoothedSal/=temp_window_size;
	smoothedSal.convertTo(smoothedSal, CV_8UC1);
	return smoothedSal;
}

int getNbLinesFile(char * File)
{
	int nbLines=0;
	std::ifstream listFramesFile(File);
	std::string line;

	if (listFramesFile.is_open())
	{
		while ( listFramesFile.good() )
		{
			listFramesFile >> line;
			nbLines++;
		}
		listFramesFile.close();
	}
	return nbLines;
}
