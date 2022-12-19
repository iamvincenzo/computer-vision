// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

void addZeroPaddingGeneral(const cv::Mat &src, const cv::Mat &krnl, cv::Mat &padded, const cv::Point anchor, bool zeroPad = true)
{
	int padHTop = anchor.x;
	int padHBottom = krnl.rows - anchor.x - 1;
	int padWLeft = anchor.y;
	int padWRight = krnl.cols - anchor.y - 1;

	if (zeroPad)
	{
		padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(0));
	}
	else // non crea problemi nel caso dell'erosione
	{
		padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(255));
	}

	for (int v = padHTop; v < padded.rows - padHBottom; ++v)
	{
		for (int u = padWLeft; u < padded.cols - padWRight; ++u)
		{
			padded.at<u_char>(v, u) = src.at<u_char>(v - padHTop, u - padWLeft);
		}
	}

	cv::namedWindow("padded image", cv::WINDOW_NORMAL);
	imshow("padded image", padded);

	return;
}

void createKernel(cv::Mat &krnl, const int ksize)
{
	// float dataKx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	// cv::Mat krnl(3, 3, CV_32F, dataKx);
	// cv::Mat Ky(3, 3, CV_32F, dataKy);

	krnl = cv::Mat(ksize, ksize, CV_8UC1, cv::Scalar(0));

	int j = 0;

	for (int v = 0; v < krnl.rows; ++v)
	{
		for (int u = 0; u < krnl.cols; ++u)
		{
			for (int k = 0; k < krnl.channels(); ++k)
			{
				j = ((u + v * krnl.cols) * krnl.elemSize() + k * krnl.elemSize1());

				if (j == 1 || j == 3 || j == 4 || j == 5 || j == 7)
				{
					krnl.data[j] = 255;
				}
			}
		}
	}

	return;
}

void myErodeBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outErodeB, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outErodeB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool diff;

	for (int v = 0; v < outErodeB.rows; ++v)
	{
		for (int u = 0; u < outErodeB.cols; ++u)
		{
			diff = false;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (krnl.data[j + i * krnl.cols] != padded.data[(u + i) + (v + j) * padded.cols])
						{
							diff = true;
							break;
						}
					}
				}

				if (diff)
					break;
			}

			if (!diff)
				outErodeB.data[(u + v * outErodeB.cols)] = 255;
		}
	}

	return;
}

void myDilateBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outDilateB, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outDilateB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool eq;

	for (int v = 0; v < outDilateB.rows; ++v)
	{
		for (int u = 0; u < outDilateB.cols; ++u)
		{
			eq = false;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (krnl.data[j + i * krnl.cols] == padded.data[(u + i) + (v + j) * padded.cols])
						{
							eq = true;
							break;
						}
					}
				}

				if (eq)
					break;
			}

			if (eq)
				outDilateB.data[(u + v * outDilateB.cols)] = 255;
		}
	}

	return;
}

void myOpenBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outOpenB, const cv::Point anchor)
{
	cv::Mat tmp;
	myErodeBinary(src, krnl, tmp, anchor);

	myDilateBinary(tmp, krnl, outOpenB, anchor);

	return;
}

void myCloseBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outCloseB, const cv::Point anchor)
{
	cv::Mat tmp;
	myDilateBinary(src, krnl, tmp, anchor);

	myErodeBinary(tmp, krnl, outCloseB, anchor);

	return;
}

void myErodeGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outErodeG, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor, false);

	outErodeG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int min;

	for (int v = 0; v < outErodeG.rows; ++v)
	{
		for (int u = 0; u < outErodeG.cols; ++u)
		{
			min = 255;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (padded.data[(u + i) + (v + j) * padded.cols] < min)
						{
							min = padded.data[(u + i) + (v + j) * padded.cols];
						}
					}
				}
			}

			outErodeG.data[(u + v * outErodeG.cols)] = min;
		}
	}

	return;
}

void myDilateGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outDilateG, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outDilateG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int max;

	for (int v = 0; v < outDilateG.rows; ++v)
	{
		for (int u = 0; u < outDilateG.cols; ++u)
		{
			max = 0;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (padded.data[(u + i) + (v + j) * padded.cols] > max)
						{
							max = padded.data[(u + i) + (v + j) * padded.cols];
						}
					}
				}
			}

			outDilateG.data[(u + v * outDilateG.cols)] = max;
		}
	}

	return;
}

void myOpenGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outOpenB, const cv::Point anchor)
{
	cv::Mat tmp;
	myErodeGrayScale(src, krnl, tmp, anchor);

	myDilateGrayScale(tmp, krnl, outOpenB, anchor);

	return;
}

void myCloseGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outCloseB, const cv::Point anchor)
{
	cv::Mat tmp;
	myDilateGrayScale(src, krnl, tmp, anchor);

	myErodeGrayScale(tmp, krnl, outCloseB, anchor);

	return;
}

bool ParseInputs(ArgumentList &args, int argc, char **argv);

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_GRAYSCALE; // cv::IMREAD_COLOR;
	// int ksize = 3;

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

		// create kernel (a croce)
		u_char dataKrnl[9] = {0, 255, 0, 255, 255, 255, 0, 255, 0};
		cv::Point anchor(1, 1);
		cv::Mat krnl(3, 3, CV_8UC1, dataKrnl);

		// Binary
		cv::Mat binaryImage;
		cv::threshold(image, binaryImage, 150, 255, cv::THRESH_BINARY);

		cv::Mat outErodeB;
		myErodeBinary(binaryImage, krnl, outErodeB, anchor);

		cv::Mat outDilateB;
		myDilateBinary(binaryImage, krnl, outDilateB, anchor);

		cv::Mat outOpenB;
		myOpenBinary(binaryImage, krnl, outOpenB, anchor);

		cv::Mat outCloseB;
		myCloseBinary(binaryImage, krnl, outCloseB, anchor);

		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		cv::namedWindow("binary image", cv::WINDOW_NORMAL);
		imshow("binary image", binaryImage);

		cv::namedWindow("erosion binary", cv::WINDOW_NORMAL);
		imshow("erosion binary", outErodeB);

		cv::namedWindow("dilate binary", cv::WINDOW_NORMAL);
		imshow("dilate binary", outDilateB);

		cv::namedWindow("open binary", cv::WINDOW_NORMAL);
		imshow("open binary", outOpenB);

		cv::namedWindow("close binary", cv::WINDOW_NORMAL);
		imshow("close binary", outCloseB);

		// Gray scale

		cv::Mat outErodeG;
		myErodeGrayScale(image, krnl, outErodeG, anchor);

		cv::Mat outDilateG;
		myDilateGrayScale(image, krnl, outDilateG, anchor);

		cv::Mat outOpenG;
		myOpenGrayScale(image, krnl, outOpenG, anchor);

		cv::Mat outCloseG;
		myCloseGrayScale(image, krnl, outCloseG, anchor);

		cv::namedWindow("erosion grayscale", cv::WINDOW_NORMAL);
		imshow("erosion grayscale", outErodeG);

		cv::namedWindow("dilate grayscale", cv::WINDOW_NORMAL);
		imshow("dilate grayscale", outDilateG);

		cv::namedWindow("open grayscale", cv::WINDOW_NORMAL);
		imshow("open grayscale", outOpenG);

		cv::namedWindow("close grayscale", cv::WINDOW_NORMAL);
		imshow("close grayscale", outCloseG);

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
