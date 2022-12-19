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

	float sum_b = 0.0, sum_f = 0.0, wb = 0.0, wf = 0.0, tot = 0.0, ub = 0.0, uf = 0.0, sigma_b = 0.0;

	std::vector<float> thSigmaB;

	std::vector<int> vecOfValues;
	vecOfValues.resize(256);
	std::fill(vecOfValues.begin(), vecOfValues.end(), 0);

	generateHistogram(src, vecOfValues);

	for (int i = 0; i < (int)vecOfValues.size(); ++i)
	{
		tot += vecOfValues[i];
	}

	for (int th = 0; th < (int)vecOfValues.size(); ++th)
	{
		sum_b = 0.0;
		sum_f = 0.0;
		wb = 0.0;
		wf = 0.0;
		ub = 0.0;
		uf = 0.0;
		sigma_b = 0.0;

		for (int i = 0; i <= (int)th; ++i)
		{
			sum_b += vecOfValues[i];
			ub += (vecOfValues[i] * i);
		}

		if (sum_b == 0)
		{
			thSigmaB.push_back(0.0);
			continue;
		}

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			sum_f += vecOfValues[i];
			uf += (vecOfValues[i] * i);
		}

		if (sum_f == 0)
		{
			thSigmaB.push_back(0.0);
			continue;
		}

		wb = sum_b / tot;
		ub = ub / sum_b;
		wf = sum_f / tot;
		uf = uf / sum_f;
		sigma_b = wb * wf * pow((ub - uf), 2);

		thSigmaB.push_back(sigma_b);
	}

	// indice del massimo nel vector
	int th = max_element(thSigmaB.begin(), thSigmaB.end()) - thSigmaB.begin();

	std::cout << "\n\noptimal-threshold: " << th << " and optimal-variance: " << thSigmaB[th] << std::endl;

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
