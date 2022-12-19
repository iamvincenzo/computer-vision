// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>

#define RGGB_BGGR 0
#define GRBG_GBRG 1

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
	int top_left_x;
	int top_left_y;
	int h;
	int w;
	int padding_size;
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

void downsample(const cv::Mat &src, cv::Mat &out, const u_char pattern)
{
	// shit indices
	int k1 = 0, l1 = 0;
	int k2 = 0, l2 = 0;
	int sum = 0;
	int mean = 0;
	int stride = 2;
	int G = 1;

	switch (pattern)
	{
	case RGGB_BGGR:
		k1 = 0;
		l1 = 1;
		k2 = 1;
		l2 = 0;
		std::cout << "RGGB o BGGR" << std::endl;
		break;
	case GRBG_GBRG:
		k1 = 0;
		l1 = 0;
		k2 = 1;
		l2 = 1;
		std::cout << "GRBG o GBRG" << std::endl;
		break;
	}

	out = cv::Mat(src.rows / 2, src.cols / 2, CV_8UC1, cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			// (u * stride) + l1) --> stride di 2 per downsample + l1/k1 = cella con G
			// (v * stride) + k1) * src.cols --> stride di 2 per downsample + l2/k2 = cella con G
			// G * src.elemSize1() --> BGR = 012 --> prende il canale G
			sum += (int)src.data[(((u * stride) + l1) + (((v * stride) + k1) * src.cols)) * src.elemSize() + G * src.elemSize1()];
			sum += (int)src.data[(((u * stride) + l2) + (((v * stride) + k2) * src.cols)) * src.elemSize() + G * src.elemSize1()];

			mean = sum / 2;
			out.data[u + v * out.cols] = mean;
			sum = 0;
			mean = 0;
		}
	}

	return;
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_COLOR;

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
				  << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

		//////////////////////
		// processing code here

		std::string pattern = args.image_name.substr(21, 4);

		cv::Mat out;

		if ((pattern.compare("RGGB") == 0) || (pattern.compare("BGGR") == 0))
			downsample(image, out, RGGB_BGGR);

		else if ((pattern.compare("GRBG") == 0) || (pattern.compare("GBRG")) == 0)
			downsample(image, out, GRBG_GBRG);

		std::string title = "out " + pattern;

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// display image
		cv::namedWindow(title, cv::WINDOW_NORMAL);
		cv::imshow(title, out);

		// /////////////////////

		// wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << "key " << int(key) << std::endl;

		// here you can implement some looping logic using key value:
		//  - pause
		//  - stop_left_x
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

	while ((c = getopt(argc, argv, "hi:t:c:p:")) != -1)
		switch (c)
		{
		case 't':
			args.wait_t = atoi(optarg);
			break;
		case 'i':
			args.image_name = optarg;
			break;
		case 'c':
			args.top_left_x = atoi(optarg);
			args.top_left_y = atoi(optarg);
			args.w = atoi(optarg);
			args.h = atoi(optarg);
			break;
		case 'p':
			args.padding_size = atoi(optarg);
		case 'h':
		default:
			std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
			std::cout << "exit:  type q" << std::endl
					  << std::endl;
			std::cout << "Allowed options:" << std::endl
					  << "   -h                       produce help message" << std::endl
					  << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
					  << "   -t arg                   wait before next frame (ms)" << std::endl
					  << "   -c arg                   crop image. Use top_left_x top_left_y w h" << std::endl
					  << "   -p arg                   pad image. Use padding_size"
					  << std::endl;
			return false;
		}
	return true;
}

#endif
