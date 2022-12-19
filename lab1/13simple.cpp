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

#define RGGB 0
#define BGGR 1
#define GRBG 2
#define GBRG 3

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

/* Immagini RGGB:

R G R G R G
G B G B G B
R G R G R G
G B G B G B

	0		1		2
0	R G --> G R --> R G
	G B	    B G     G B

1	G B --> B G --> G B
	R G     G R	    R G

--> RGGB if (src.rows % 2 == 0 && src.cols % 2 == 0)
--> GRBG if (src.rows % 2 == 0 && src.cols % 2 != 0)
--> GBRG if (src.rows % 2 != 0 && src.cols % 2 == 0)
--> BGGR if (src.rows % 2 != 0 && src.cols % 2 != 0)

*/

/* Immagini BGGR:

B G B G B G
G R G R G R
B G B G B G
G R G R G R

	0		1		2
0	B G --> G B --> B G
	G R     R G     G R

1	G R --> R G --> G R
	B G     G B     B G

--> BGGR if (src.rows % 2 == 0 && src.cols % 2 == 0)
--> GBRG if (src.rows % 2 == 0 && src.cols % 2 != 0)
--> GRBG if (src.rows % 2 != 0 && src.cols % 2 == 0)
--> RGGB if (src.rows % 2 != 0 && src.cols % 2 != 0)

*/

/* Immagini GRBG:

G R G R G R
B G B G B G
G R G R G R
B G B G B G

	0		1		2
0	G R --> R G --> G R
	B G     G B     B G

1	B G --> G B --> B G
	G R     R G     G R

--> GRBG if (src.rows % 2 == 0 && src.cols % 2 == 0)
--> RGGB if (src.rows % 2 == 0 && src.cols % 2 != 0)
--> BGGR if (src.rows % 2 != 0 && src.cols % 2 == 0)
--> GBRG if (src.rows % 2 != 0 && src.cols % 2 != 0)

*/

/* Immagini GBRG

G B G B G B
R G R G R G
G B G B G B
R G R G R G

	0		1		2
0	G B --> B G --> G B
	R G     G R     R G

1	R G --> G R --> R G
	G B     B G     G B

--> GBRG if (src.rows % 2 == 0 && src.cols % 2 == 0)
--> BGGR if (src.rows % 2 == 0 && src.cols % 2 != 0)
--> RGGB if (src.rows % 2 != 0 && src.cols % 2 == 0)
--> GRBG if (src.rows % 2 != 0 && src.cols % 2 != 0)

*/

void simple(const cv::Mat &src, cv::Mat &out, const u_char pattern)
{
	// shift indices
	int kg1 = 0, lg1 = 0, kg2 = 0, lg2 = 0, kr = 0, lr = 0, kb = 0, lb = 0;
	int chanB = 0, chanG = 1, chanR = 2;
	int B = 0, G1 = 0, G2 = 0, R = 0;

	out = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if ((pattern == RGGB && (v % 2 == 0 && u % 2 == 0)) ||
				(pattern == BGGR && (v % 2 != 0 && u % 2 != 0)) ||
				(pattern == GRBG && (v % 2 == 0 && u % 2 != 0)) ||
				(pattern == GBRG && (v % 2 != 0 && u % 2 == 0)))
			{
				// RGGB shift
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 0;
				lr = 0;
				kb = 1;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 != 0 && u % 2 != 0)))
			{
				// GRBG shift
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 0;
				lr = 1;
				kb = 1;
				lb = 0;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 == 0)))
			{
				// GBRG shift
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 1;
				lr = 0;
				kb = 0;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 != 0)))
			{
				// BGGR shift
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 1;
				lr = 1;
				kb = 0;
				lb = 0;
			}

			B = (int)src.data[((u + lb) + ((v + kb) * src.cols)) * src.elemSize() + chanB * src.elemSize1()];
			G1 = (int)src.data[((u + lg1) + ((v + kg1) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];
			G2 = (int)src.data[((u + lg2) + ((v + kg2) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];
			R = (int)src.data[((u + lr) + ((v + kr) * src.cols)) * src.elemSize() + chanR * src.elemSize1()];

			out.data[(u + v * out.cols) * out.elemSize() + chanB * out.elemSize1()] = B;
			out.data[(u + v * out.cols) * out.elemSize() + chanG * out.elemSize1()] = (G1 + G2) / 2;
			out.data[(u + v * out.cols) * out.elemSize() + chanR * out.elemSize1()] = R;
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

		if (pattern.compare("RGGB") == 0)
			simple(image, out, RGGB);

		else if (pattern.compare("BGGR") == 0)
			simple(image, out, BGGR);

		else if (pattern.compare("GRBG") == 0)
			simple(image, out, GRBG);

		else if (pattern.compare("GBRG") == 0)
			simple(image, out, GBRG);

		std::string title = "out " + pattern;

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// display image
		cv::namedWindow(title, cv::WINDOW_NORMAL);
		cv::imshow(title, out);

		/////////////////////

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

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
