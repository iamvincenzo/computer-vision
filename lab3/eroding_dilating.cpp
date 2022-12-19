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

bool checkOddSquareKrn(const cv::Mat &krn)
{
	if (krn.rows == krn.cols && krn.rows % 2 != 0 && krn.cols % 2 != 0)
		return true;
	else
		return false;
}

void addZeroPadding(const cv::Mat &src, cv::Mat &padded, const int padH, const int padW, bool zeroPad = true)
{
	/**
	 * Per immagini binarie
	 *
	 * padded_height = (input height + padding height top + padding height bottom)
	 * padded_width = (input width + padding width right + padding width left)
	 */
	if (zeroPad)
	{
		padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(0));
	}
	else 
	{
		padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(255));
	}

	for (int v = padH; v < padded.rows - padH; ++v)
	{
		for (int u = padW; u < padded.cols - padW; ++u)
		{
			for (int k = 0; k < padded.channels(); ++k)
			{
				padded.data[((u + v * padded.cols) * padded.elemSize() + k * padded.elemSize1())] = src.data[(((u - padW) + (v - padH) * src.cols) * src.elemSize() + k * src.elemSize1())];
			}
		}
	}

	return;
}

void createKernel(cv::Mat &krn, const int ksize)
{
	krn = cv::Mat(ksize, ksize, CV_8UC1, cv::Scalar(0));

	int j = 0;

	for (int v = 0; v < krn.rows; ++v)
	{
		for (int u = 0; u < krn.cols; ++u)
		{
			for (int k = 0; k < krn.channels(); ++k)
			{
				j = ((u + v * krn.cols) * krn.elemSize() + k * krn.elemSize1());

				if (j == 1 || j == 3 || j == 4 || j == 5 || j == 7)
				{
					krn.data[j] = 255;
				}
			}
		}
	}

	return;
}

void myErodeBinary(cv::Mat &src, cv::Mat &krn, cv::Mat &outErodeB)
{
	if (!checkOddSquareKrn(krn))
	{
		std::cout << "ERRORE: Il kernel deve essere dispari e quadrato" << std::endl;
		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	outErodeB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool diff;

	for (int v = 0; v < outErodeB.rows; ++v)
	{
		for (int u = 0; u < outErodeB.cols; ++u)
		{
			diff = false;

			for (int i = 0; i < krn.rows; ++i)
			{
				for (int j = 0; j < krn.cols; ++j)
				{
					if (krn.data[j + i * krn.cols] == 255)
					{
						if (krn.data[j + i * krn.cols] != padded.data[(u + i) + (v + j) * padded.cols])
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

void myDilateBinary(cv::Mat &src, cv::Mat &krn, cv::Mat &outDilateB)
{
	if (!checkOddSquareKrn(krn))
	{
		std::cout << "ERRORE: Il kernel deve essere dispari e quadrato" << std::endl;
		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	outDilateB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool eq;

	for (int v = 0; v < outDilateB.rows; ++v)
	{
		for (int u = 0; u < outDilateB.cols; ++u)
		{
			eq = false;

			for (int i = 0; i < krn.rows; ++i)
			{
				for (int j = 0; j < krn.cols; ++j)
				{
					if (krn.data[j + i * krn.cols] == 255)
					{
						if (krn.data[j + i * krn.cols] == padded.data[(u + i) + (v + j) * padded.cols])
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

void myOpenBinary(cv::Mat &src, cv::Mat &krn, cv::Mat &outOpenB)
{
	cv::Mat tmp;
	myErodeBinary(src, krn, tmp);

	myDilateBinary(tmp, krn, outOpenB);

	return;
}

void myCloseBinary(cv::Mat &src, cv::Mat &krn, cv::Mat &outCloseB)
{
	cv::Mat tmp;
	myDilateBinary(src, krn, tmp);

	myErodeBinary(tmp, krn, outCloseB);

	return;
}

void myErodeGrayScale(cv::Mat &src, cv::Mat &krn, cv::Mat &outErodeG)
{
	if (!checkOddSquareKrn(krn))
	{
		std::cout << "ERRORE: Il kernel deve essere dispari e quadrato" << std::endl;
		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW, false);

	outErodeG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int min;

	for (int v = 0; v < outErodeG.rows; ++v)
	{
		for (int u = 0; u < outErodeG.cols; ++u)
		{
			min = 255;

			for (int i = 0; i < krn.rows; ++i)
			{
				for (int j = 0; j < krn.cols; ++j)
				{
					if (krn.data[j + i * krn.cols] == 255)
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

void myDilateGrayScale(cv::Mat &src, cv::Mat &krn, cv::Mat &outDilateG)
{
	if (!checkOddSquareKrn(krn))
	{
		std::cout << "ERRORE: Il kernel deve essere dispari e quadrato" << std::endl;
		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	outDilateG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int max;

	for (int v = 0; v < outDilateG.rows; ++v)
	{
		for (int u = 0; u < outDilateG.cols; ++u)
		{
			max = 0;

			for (int i = 0; i < krn.rows; ++i)
			{
				for (int j = 0; j < krn.cols; ++j)
				{
					if (krn.data[j + i * krn.cols] == 255)
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

void myOpenGrayScale(cv::Mat &src, cv::Mat &krn, cv::Mat &outOpenB)
{
	cv::Mat tmp;
	myErodeGrayScale(src, krn, tmp);

	myDilateGrayScale(tmp, krn, outOpenB);

	return;
}

void myCloseGrayScale(cv::Mat &src, cv::Mat &krn, cv::Mat &outCloseB)
{
	cv::Mat tmp;
	myDilateGrayScale(src, krn, tmp);

	myErodeGrayScale(tmp, krn, outCloseB);

	return;
}

bool ParseInputs(ArgumentList &args, int argc, char **argv);

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_GRAYSCALE; // cv::IMREAD_COLOR;
	int ksize = 3;

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

		// Binary

		cv::Mat krn;
		createKernel(krn, ksize);

		cv::Mat binaryImage;
		cv::threshold(image, binaryImage, 150, 255, cv::THRESH_BINARY);

		cv::Mat outErodeB;
		myErodeBinary(binaryImage, krn, outErodeB);

		cv::Mat outDilateB;
		myDilateBinary(binaryImage, krn, outDilateB);

		cv::Mat outOpenB;
		myOpenBinary(binaryImage, krn, outOpenB);

		cv::Mat outCloseB;
		myCloseBinary(binaryImage, krn, outCloseB);

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
		myErodeGrayScale(image, krn, outErodeG);

		cv::Mat outDilateG;
		myDilateGrayScale(image, krn, outDilateG);

		cv::Mat outOpenG;
		myOpenGrayScale(image, krn, outOpenG);

		cv::Mat outCloseG;
		myCloseGrayScale(image, krn, outCloseG);

		cv::namedWindow("erosion grayscale", cv::WINDOW_NORMAL);
		imshow("erosion grayscale", outErodeG);

		cv::namedWindow("dilate grayscale", cv::WINDOW_NORMAL);
		imshow("dilate grayscale", outDilateG);

		cv::namedWindow("open grayscale", cv::WINDOW_NORMAL);
		imshow("open grayscale", outOpenG);

		cv::namedWindow("close grayscale", cv::WINDOW_NORMAL);
		imshow("close grayscale", outCloseG);

		/*
			// Create a structuring element (SE)
			int morph_size = 2;
			cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
			cv::Mat erod, dill;

			// For Erosion
			cv::erode(image, erod, element, cv::Point(-1, -1), 1);

			// For Dilation
			cv::dilate(image, dill, element, cv::Point(-1, -1), 1);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			cv::namedWindow("erosion", cv::WINDOW_NORMAL);
			imshow("erosion", erod);

			cv::namedWindow("dilate", cv::WINDOW_NORMAL);
			imshow("dilate", dill);
		*/

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
