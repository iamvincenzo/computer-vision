// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

bool checkOddKernel(const cv::Mat &krn)
{
	if (krn.cols % 2 != 0 && krn.rows % 2 != 0)
		return true;
	else
		return false;
}

void addZeroPadding(const cv::Mat &src, cv::Mat &padded, const int padH, const int padW)
{
	padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(0));

	for (int v = padH; v < padded.rows - padH; ++v)
	{
		for (int u = padW; u < padded.cols - padW; ++u)
		{
			padded.at<u_char>(v, u) = src.at<u_char>((v - padH), (u - padW));
		}
	}

	// display image
	cv::namedWindow("padded image", cv::WINDOW_NORMAL);
	cv::imshow("padded image", padded);

	return;
}

void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stride = 1)
{
	if (!checkOddKernel(krn))
	{
		std::cout << "ERRORE: il kernel deve essere dispari e quadrato." << std::endl;

		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	/**
	 * output_height = (int) ((input height + padding height top + padding height bottom - kernel height) / (stride height) + 1)
	 * output_width = (int) ((input width + padding width right + padding width left - kernel width) / (stride width) + 1)
	 */
	out = cv::Mat((int)((src.rows + 2 * padH - krn.rows) / stride) + 1, (int)((src.cols + 2 * padW - krn.cols) / stride) + 1, CV_32SC1);

	float g_kl;
	float w_sum;

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			w_sum = 0.0;

			for (int k = 0; k < krn.rows; ++k)
			{
				for (int l = 0; l < krn.cols; ++l)
				{
					g_kl = krn.at<float>(k, l);
					w_sum += g_kl * (float)padded.at<u_char>((v * stride) + k, (u * stride) + l);
					// w_sum += g_kl * (float)padded.at<u_char>((v + k) * stride, (u + l) * stride);
				}
			}

			out.at<int32_t>(v, u) = w_sum;
		}
	}

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myfilter2D conv", cv::WINDOW_NORMAL);
	cv::imshow("myfilter2D conv", outDisplay);

	return;
}

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int ksize = 3;
	int stride = 1;

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

		cv::Mat image = cv::imread(frame_name);
		if (image.empty())
		{
			std::cout << "Unable to open " << frame_name << std::endl;
			return 1;
		}

		//////////////////////
		// processing code here

		cv::Mat grey;
		cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY); // conversione dell'immagine da RGB a scala di grigi e salva il risultato in grey

		// custom convolution
		cv::Mat myfilter2Dresult;
		cv::Mat custom_kernel(ksize, ksize, CV_32FC1, 1.0 / (ksize * ksize));
		myfilter2D(grey, custom_kernel, myfilter2Dresult);

		cv::Mat custom_blurred;
		cv::filter2D(grey, custom_blurred, CV_32F, custom_kernel);
		cv::convertScaleAbs(custom_blurred, custom_blurred);

		// display opencv convolution result
		cv::namedWindow("opencv conv", cv::WINDOW_NORMAL);
		cv::imshow("opencv conv", custom_blurred);

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// display image greyscale
		cv::namedWindow("grey", cv::WINDOW_NORMAL);
		cv::imshow("grey", grey);

		//////////////////////

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
		case 's':
			if (stride != 1)
				--stride;
			std::cout << "Stride: " << stride << std::endl;
			break;
		case 'S':
			++stride;
			std::cout << "Stride: " << stride << std::endl;
			break;

		case 'c':
			cv::destroyAllWindows();
			break;
		case 'p':
			std::cout << "Mat = " << std::endl
					  << image << std::endl;
			break;
		case 'k':
		{
			static int sindex = 0;
			int values[] = {3, 5, 7, 11, 13};
			ksize = values[++sindex % 5];
			std::cout << "Setting Kernel size to: " << ksize << std::endl;
		}
		break;
		case 'g':
			break;
		case 'q':
			exit(0);
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
	args.wait_t = 0;

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
