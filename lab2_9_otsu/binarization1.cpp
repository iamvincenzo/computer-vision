// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <cmath>

#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>  //std::numeric_limits
#include <numeric> //std::accumulate

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

/**
 * @brief
 *
 * @param src
 * @param vecOfValues
 */
void generateHistogram(const cv::Mat &src, std::vector<int> &vecOfValues)
{
	for (int v = 0; v < src.rows; ++v)
	{
		for (int u = 0; u < src.cols; ++u)
		{
			// ++vecOfValues[(int)src.data[((u + v * src.cols) * src.elemSize())]];
			++vecOfValues[(int)src.at<u_char>(v, u)];
		}
	}

	return;
}

void getMinIndexValue(const std::vector<float> &v, int &index, float &val)
{
	index = 0;
	val = 4000.0;

	for (int i = 0; i < (int)v.size(); ++i)
	{
		if (v[i] > 0 && v[i] < val)
		{
			val = v[i];
			index = i;
		}
	}
}

/**
 * @brief
 *
 * @param src
 * @param out
 * @param vecOfValues
 */
void outsuMethod(const cv::Mat &src, cv::Mat &out)
{
	//////////////////////
	// th - computation

	float pBg = 0.0, pFg = 0.0, wBg = 0.0, wFg = 0.0, pAll = 0.0, varBg = 0.0, varFg = 0.0, varTot = 0.0,
		  sumBg = 0.0, sumFg = 0.0, meanXBg = 0.0, meanXFg = 0.0, sumVarBg = 0.0, sumVarFg = 0.0;

	std::vector<float> varTotVec;

	std::vector<int> vecOfValues;
	vecOfValues.resize(256);
	std::fill(vecOfValues.begin(), vecOfValues.end(), 0);

	generateHistogram(src, vecOfValues);

	for (int i = 0; i < (int)vecOfValues.size(); ++i)
	{
		pAll += vecOfValues[i]; // the total count of pixels in an image
	}

	for (int th = 0; th < (int)vecOfValues.size(); ++th)
	{
		pBg = 0.0;
		pFg = 0.0;
		wBg = 0.0;
		wFg = 0.0;
		varBg = 0.0;
		varFg = 0.0;
		varTot = 0.0;
		sumBg = 0.0;
		sumFg = 0.0;
		meanXBg = 0.0;
		meanXFg = 0.0;
		sumVarBg = 0.0;
		sumVarFg = 0.0;

		for (int i = 0; i <= (int)th; ++i)
		{
			pBg += vecOfValues[i]; // the count of background pixels at threshold th
			sumBg += (vecOfValues[i] * i);
		}

		if (pBg == 0)
		{
			varTotVec.push_back(0.0);
			continue;
		}

		meanXBg = sumBg / pBg;

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			pFg += vecOfValues[i]; // the count of foreground pixels at threshold th
			sumFg += (vecOfValues[i] * i);
		}

		if (pFg == 0)
		{
			varTotVec.push_back(0.0);
			continue;
		}

		meanXFg = sumFg / pFg;

		wBg = pBg / pAll; // weight for background

		for (int i = 0; i <= (int)th; ++i)
		{
			sumVarBg += vecOfValues[i] * pow((i - meanXBg), 2);
		}

		varBg = sumVarBg / pBg; // (pBg - 1);

		wFg = pFg / pAll; // weight for foreground

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			sumVarFg += vecOfValues[i] * pow((i - meanXFg), 2);
		}

		varFg = sumVarFg / pFg; // (pFg - 1);

		varTot = wBg * varBg + wFg * varFg;

		varTotVec.push_back(varTot);

		// if (th % 100 == 0)
		// {
		// 	std::cout << "th " << th << " - meanXBg: " << meanXBg << std::endl;
		// 	std::cout << "th " << th << " - meanXFg: " << meanXFg << std::endl;
		// 	std::cout << "th " << th << "- wBg: " << wBg << std::endl;
		// 	std::cout << "th " << th << "- wBg: " << wFg << std::endl;
		// 	std::cout << "th " << th << " - varBg: " << varBg << std::endl;
		// 	std::cout << "th " << th << " - varFg: " << varFg << std::endl;
		// 	std::cout << "th " << th << " - varTot: " << varTot << std::endl;
		// }
	}

	// for (auto &elem : varTotVec)
	// {
	// 	std::cout << "varTot: " << elem << std::endl;
	// }

	// indice del massimo nel vector
	// int th = max_element(varTotVec.begin(), varTotVec.end()) - varTotVec.begin();

	int th = 0;
	float minVal = 0.0;
	getMinIndexValue(varTotVec, th, minVal);

	std::cout << "\n\noptimal-threshold: " << th << " and optimal-variance: " << varTotVec[th] << std::endl;

	//////////////////////

	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if (src.data[((u + v * src.cols) * src.elemSize())] >= th)
			{
				out.data[((u + v * out.cols) * out.elemSize())] = 255;
			}
		}
	}

	return;
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_GRAYSCALE; // cv::IMREAD_COLOR;

	std::cout << "Simple program." << std::endl;

	//////////////////////
	// parse argument list:
	//////////////////////
	ArgumentList args;
	if (!ParseInputs(args, argc, argv))
	{
		exit(0);
	}

	while (!exit_loop)
	{
		// generating file name
		//
		// multi frame case
		if (args.image_name.find('%') != std::string::npos)
			sprintf(frame_name, (const char *)(args.image_name.c_str()), frame_number);
		else // single frame case
			sprintf(frame_name, "%s", args.image_name.c_str());

		// opening file
		std::cout << "Opening " << frame_name << std::endl;

		cv::Mat image = cv::imread(frame_name, imreadflags);
		if (image.empty())
		{
			std::cout << "Unable to open " << frame_name << std::endl;
			return 1;
		}

		std::cout << "The image has " << image.channels() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
				  << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl
				  << std::endl;

		//////////////////////
		// processing code here

		// cv::Mat src(4, 4, CV_8UC1, cv::Scalar(0));

		// std::vector<int> tmp = {120, 120, 21, 22, 25, 26, 27, 160, 180, 190, 123, 145, 165, 175, 23, 24};
		// int i = 0;

		// for (int v = 0; v < src.rows; ++v)
		// {
		// 	for (int u = 0; u < src.cols; ++u)
		// 	{
		// 		src.data[(u + v * src.cols) * src.elemSize()] = tmp[i];
		// 		i++;
		// 	}
		// }

		std::cout << std::endl
				  << std::endl
				  << "Before binarization: " << std::endl
				  << std::endl;

		for (int v = 0; v < image.rows; ++v)
		{
			for (int u = 0; u < image.cols; ++u)
			{
				if (image.data[(u + v * image.cols) * image.elemSize()] != 0 &&
					image.data[(u + v * image.cols) * image.elemSize()] != 255)
				{
					std::cout << (int)image.data[(u + v * image.cols) * image.elemSize()] << " ";
				}
			}
		}

		cv::Mat out;
		outsuMethod(image, out);
		// outsuMethod(src, out);

		std::cout << std::endl
				  << std::endl
				  << "After binarization: " << std::endl
				  << std::endl;

		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				if (out.data[(u + v * out.cols) * out.elemSize()] != 0 &&
					out.data[(u + v * out.cols) * out.elemSize()] != 255)
				{
					std::cout << (int)out.data[(u + v * out.cols) * out.elemSize()] << std::endl;
				}
			}
		}

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// display image
		cv::namedWindow("binary out", cv::WINDOW_NORMAL);
		cv::imshow("binary out", out);

		/////////////////////

		// wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << "key " << int(key) << std::endl;

		// here you can implement some looping logic using key value:
		//  - pause
		//  - stop
		//  - step back
		//  - step forward
		//  - loop on the same frame

		switch (key)
		{
		case 'p':
			std::cout << "Mat = " << std::endl
					  << image << std::endl;
			break;
		case 'q':
			exit_loop = 1;
			break;
		case 'c':
			std::cout << "SET COLOR imread()" << std::endl;
			imreadflags = cv::IMREAD_COLOR;
			break;
		case 'g':
			std::cout << "SET GREY  imread()" << std::endl;
			imreadflags = cv::IMREAD_GRAYSCALE; // Y = 0.299 R + 0.587 G + 0.114 B
			break;
		}

		frame_number++;
	}

	return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}
#else

#include <unistd.h>
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

#endif
