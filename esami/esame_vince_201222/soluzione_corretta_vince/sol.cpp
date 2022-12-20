// VINCENZO FRAELLO 339641

// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

// prototipi di funzioni

bool ParseInputs(ArgumentList &args, int argc, char **argv);

void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img);

void myHoughTransfLines(const cv::Mat &src, cv::Mat &accumulator, const int minTheta, const int maxTheta, const int diagonal);

void myFindPeaks3x3(const cv::Mat &src, cv::Mat &out);

// programma principale

int main(int argc, char **argv)
{
	int imreadflags = cv::IMREAD_GRAYSCALE;

	std::cout << "Simple program." << std::endl;

	//////////////////////
	// parse argument list:
	//////////////////////
	ArgumentList args;
	if (!ParseInputs(args, argc, argv))
	{
		exit(0);
	}

	// opening file
	std::cout << "Opening " << args.image_name << std::endl;

	cv::Mat image = cv::imread(args.image_name.c_str(), imreadflags);
	if (image.empty())
	{
		std::cout << "Unable to open " << args.image_name << std::endl;
		return 1;
	}

	//////////////////////
	// processing code here

	// display image
	cv::namedWindow("original image", cv::WINDOW_NORMAL);
	cv::imshow("original image", image);

	cv::Mat blurred, edges;
	cv::GaussianBlur(image, blurred, cv::Size(3, 3), 3);
	cv::Canny(blurred, edges, 50, 150);

	// display edges
	cv::namedWindow("CANNY", cv::WINDOW_NORMAL);
	cv::imshow("CANNY", edges);

	// YOUR CODE HERE: COMPUTE ACCUMULATOR
	/*******************************/
	int minTheta = 0;
	int maxTheta = 360;
	int threshold = 200;

	int diagonal = pow(pow(image.rows, 2) + pow(image.cols, 2), 0.5); // diagonale dell'immagine nel caso di immagini non quadrate
	/*******************************/

	cv::Mat accumulator = cv::Mat::zeros(maxTheta - minTheta, diagonal, CV_32SC1);

	/*******************************/
	myHoughTransfLines(edges, accumulator, minTheta, maxTheta, diagonal);
	/*******************************/

	// display accumulator
	cv::Mat accDisplay;
	cv::convertScaleAbs(accumulator, accDisplay);
	cv::namedWindow("Hough accumulator", cv::WINDOW_NORMAL);
	cv::imshow("Hough accumulator", accDisplay);

	// YOUR CODE HERE: NON MAXIMA SUPPRESSION FOR ACCUMULATOR (ignore borders maybe)
	/*******************************/
	cv::Mat nms;
	myFindPeaks3x3(accumulator, nms);
	/*******************************/

	cv::Mat accDisplayNms;
	cv::convertScaleAbs(nms, accDisplayNms);
	cv::namedWindow("Hough accumulator nms", cv::WINDOW_NORMAL);
	cv::imshow("Hough accumulator nms", accDisplayNms);

	// post NMS we convert again to 0 - 255 and apply a threshold
	cv::threshold(accDisplayNms, accDisplayNms, 180, 255, cv::THRESH_BINARY);

	// image on which we can draw lines
	cv::Mat color;
	cv::cvtColor(image, color, cv::COLOR_BGR2RGB);

	// YOUR CODE HERE draw lines
	/*******************************/
	cv::Point pt1, pt2;

	for (int t = 0; t < accDisplayNms.rows; ++t)
	{
		for (int r = 0; r < accDisplayNms.cols; ++r)
		{
			if (accDisplayNms.at<u_char>(t, r) >= threshold)
			{
				myPolarToCartesian(r, t, pt1, pt2, accDisplayNms.rows, edges);
				cv::line(color, pt1, pt2, cv::Scalar(0, 0, 88), 1, cv::LINE_AA);
			}
		}
	}

	// display image
	cv::namedWindow("lines", cv::WINDOW_NORMAL);
	cv::imshow("lines", color);
	/*******************************/

	/////////////////////

	// wait for Q key or timeout
	unsigned char c;
	while ((c = cv::waitKey(args.wait_t)) != 'q' && c != 'Q')
		;

	return 0;
}

// implementazione funzioni

bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, "hi:t:")) != -1)
		switch (c)
		{
		case 't':
			args.wait_t = atoi(optarg);
			break;
		case 'i':
			args.image_name = optarg;
			break;
		case 'h':
		default:
			std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
			std::cout << "exit:  type q" << std::endl
					  << std::endl;
			std::cout << "Allowed options:" << std::endl
					  << "   -h                       produce help message" << std::endl
					  << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
					  << "   -t arg                   wait before next frame (ms)" << std::endl
					  << std::endl;
			return false;
		}

	return true;
}

void myHoughTransfLines(const cv::Mat &src, cv::Mat &accumulator, const int minTheta, const int maxTheta, const int diagonal)
{
	if (minTheta < 0 || minTheta >= maxTheta || maxTheta <= minTheta || maxTheta > 360)
	{
		std::cerr << "Errore: valore di minTheta/maxTheta non accettabile." << std::endl;
		return;
	}

	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			if (src.at<u_char>(r, c) > 0)
			{
				for (int theta = minTheta; theta < maxTheta; ++theta)
				{
					int rho = c * cos(theta * CV_PI / 180) + r * sin(theta * CV_PI / 180);

					accumulator.at<int32_t>(theta, rho + diagonal) += 1;
				}
			}
		}
	}

	return;
}

void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img)
{
	double a = cos(theta * CV_PI / 180), b = sin(theta * CV_PI / 180);

	double x0 = a * rho, y0 = b * rho;

	p1.x = cvRound(x0 + 1000 * (-b));
	p1.y = cvRound(y0 + 1000 * (a));
	p2.x = cvRound(x0 - 1000 * (-b));
	p2.y = cvRound(y0 - 1000 * (a));

	return;
}

void myFindPeaks3x3(const cv::Mat &src, cv::Mat &out)
{
	// Non Maxima Suppression
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	for (int v = 1; v < src.rows - 1; ++v)
	{
		for (int u = 1; u < src.cols - 1; ++u)
		{
			int theta = src.at<int32_t>(v, u);

			if (src.at<int32_t>(v, u + 1) > theta ||
				src.at<int32_t>(v, u - 1) > theta ||
				src.at<int32_t>(v + 1, u - 1) > theta ||
				src.at<int32_t>(v - 1, u + 1) > theta ||
				src.at<int32_t>(v + 1, u) > theta ||
				src.at<int32_t>(v - 1, u) > theta ||
				src.at<int32_t>(v - 1, u - 1) > theta ||
				src.at<int32_t>(v + 1, u + 1) > theta)
			{
				out.at<int32_t>(v, u) = 0;
				continue;
			}

			out.at<int32_t>(v, u) = theta;
		}
	}

	return;
}
