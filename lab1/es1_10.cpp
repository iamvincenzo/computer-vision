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

		// ESERCIZIO 1

		// char channels[3] = {'R', 'G', 'B'};
		// int k = 0;

		// for (size_t j = 0; j <= 2 * image.elemSize1(); j += image.elemSize1())
		// {
		// 	std::cout << "Channel - " << channels[k] << ": " << std::endl;
		// 	++k;

		// 	for (size_t i = 0; i < image.rows * image.cols * image.elemSize(); ++i)
		// 	{
		// 		std::cout << (int)image.data[i + j] << " ";
		// 	}

		// 	std::cout << std::endl;
		// }

		// /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 2

		cv::Mat out(image.rows / 2, image.cols / 2, image.type());

		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					out.data[((u + v * out.cols) * out.elemSize() + k * out.elemSize1())] = image.data[((u * 2 + v * image.cols * 2) * image.elemSize() + k * image.elemSize1())];
				}
			}
		}

		cv::namedWindow("subsample all 2x", cv::WINDOW_NORMAL);
		cv::imshow("subsample all 2x", out);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 3

		cv::Mat out1(image.rows / 2, image.cols, image.type());

		for (int v = 0; v < out1.rows; ++v)
		{
			for (int u = 0; u < out1.cols; ++u)
			{
				for (int k = 0; k < out1.channels(); ++k)
				{
					out1.data[((u + v * out1.cols) * out1.elemSize() + k * out1.elemSize1())] = image.data[((u + v * image.cols * 2) * image.elemSize() + k * image.elemSize1())];
				}
			}
		}

		cv::namedWindow("subsample row 2x", cv::WINDOW_NORMAL);
		cv::imshow("subsample row 2x", out1);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 4

		cv::Mat out2(image.rows, image.cols / 2, image.type());

		for (int v = 0; v < out2.rows; ++v)
		{
			for (int u = 0; u < out2.cols; ++u)
			{
				for (int k = 0; k < out2.channels(); ++k)
				{
					out2.data[((u + v * out2.cols) * out2.elemSize() + k * out2.elemSize1())] = image.data[((u * 2 + v * image.cols) * image.elemSize() + k * image.elemSize1())];
				}
			}
		}

		cv::namedWindow("subsample col 2x", cv::WINDOW_NORMAL);
		cv::imshow("subsample col 2x", out2);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 5

		cv::Mat out3(image.rows, image.cols, image.type());

		for (int v = 0; v < out3.rows; ++v)
		{
			for (int u = 0; u < out3.cols; ++u)
			{
				for (int k = 0; k < out3.channels(); ++k)
				{
					out3.data[((u + v * out3.cols) * out3.elemSize() + k * out3.elemSize1())] = image.data[(((image.cols - 1 - u) + v * image.cols) * image.elemSize() + k * image.elemSize1())];
				}
			}
		}

		cv::namedWindow("horizontally flip", cv::WINDOW_NORMAL);
		cv::imshow("horizontally flip", out3);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 6

		cv::Mat out4(image.rows, image.cols, image.type());

		for (int v = 0; v < out4.rows; ++v)
		{
			for (int u = 0; u < out4.cols; ++u)
			{
				for (int k = 0; k < out4.channels(); ++k)
				{
					out4.data[((u + v * out4.cols) * out4.elemSize() + k * out4.elemSize1())] = image.data[((u + (image.rows - 1 - v) * image.cols) * image.elemSize() + k * image.elemSize1())];
				}
			}
		}

		cv::namedWindow("vertically flip", cv::WINDOW_NORMAL);
		cv::imshow("vertically flip", out4);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 7

		args.top_left_x = 50; // utente???
		args.top_left_y = 50;
		args.w = 200;
		args.h = 200;

		// verifica di essere all'interno dell'immagine di partenza
		if ((args.top_left_y) >= 0 && (args.top_left_y) <= image.rows - 1 && (args.top_left_x) >= 0 && (args.top_left_x) <= image.cols - 1)
		{
			cv::Mat out5 = cv::Mat(image, cv::Rect(args.top_left_x, args.top_left_y, args.w, args.h));

			cv::namedWindow("crop", cv::WINDOW_NORMAL);
			cv::imshow("crop", out5);
		}

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 7bis

		srand(time(NULL));

		args.top_left_x = rand() % image.cols;
		args.top_left_y = rand() % image.rows;
		args.w = rand() % image.cols + 1;
		args.h = rand() % image.rows + 1;

		cv::Mat out6 = cv::Mat::zeros(args.h, args.w, image.type());

		for (int v = 0; v < out6.rows; ++v)
		{
			for (int u = 0; u < out6.cols; ++u)
			{
				// verifica di essere all'interno dell'immagine di partenza
				if ((v + args.top_left_y) >= 0 && (v + args.top_left_y) <= image.rows - 1 && (u + args.top_left_x) >= 0 && (u + args.top_left_x) <= image.cols - 1)
				{
					for (int k = 0; k < out6.channels(); ++k)
					{
						out6.data[((u + v * out6.cols) * out6.elemSize() + k * out6.elemSize1())] = image.data[(((u + args.top_left_x) + (v + args.top_left_y) * image.cols) * image.elemSize() + k * image.elemSize1())];
					}
				}
			}
		}

		cv::namedWindow("crop rand", cv::WINDOW_NORMAL);
		cv::imshow("crop rand", out6);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 8

		args.padding_size = 10; // utente???

		cv::Mat out7 = cv::Mat::zeros(image.rows + 2 * args.padding_size, image.cols + 2 * args.padding_size, image.type());

		for (int v = args.padding_size; v < out7.rows - args.padding_size; ++v)
		{
			for (int u = args.padding_size; u < out7.cols - args.padding_size; ++u)
			{
				for (int k = 0; k < out7.channels(); ++k)
				{
					out7.data[(u + v * out7.cols) * out7.elemSize() + k * out7.elemSize1()] = image.data[((u - args.padding_size) + (v - args.padding_size) * image.cols) * image.elemSize() + k * image.elemSize1()];
				}
			}
		}

		cv::namedWindow("padding", cv::WINDOW_NORMAL);
		cv::imshow("padding", out7);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 9

		std::vector<std::vector<int>> imgs = {{0, 0}, {0, image.cols / 2}, {image.rows / 2, 0}, {image.rows / 2, image.cols / 2}};

		srand(unsigned(time(NULL)));
		std::random_shuffle(imgs.begin(), imgs.end());

		cv::Mat out8(image.rows, image.cols, image.type());

		// ciclo sui blocchi
		for (int br = 0; br < 2; ++br)
		{
			for (int bc = 0; bc < 2; ++bc)
			{
				// ogni blocco è largo cols/2 e altro rows/2 indipendentemente dal blocco
				for (int v = 0; v < image.rows / 2; ++v)
				{
					for (int u = 0; u < image.cols / 2; ++u)
					{
						for (int k = 0; k < out8.channels(); ++k)
						{
							// dato un blocco (br, bc) la sua posizione top-left nell'immagine destinazione è data da
							int dest_r = br * image.rows / 2;
							int dest_c = bc * image.cols / 2;

							// dato un blocco (br, bc) dell'immagine destinazione, la posizione nell'immagine originale top-left è data da
							int origin_r = imgs[br * 2 + bc][0];
							int origin_l = imgs[br * 2 + bc][1];

							out8.data[((u + dest_c) + (v + dest_r) * image.cols) * out8.elemSize() + k * out8.elemSize1()] = image.data[((u + origin_l) + (v + origin_r) * image.cols) * image.elemSize() + k * image.elemSize1()];
						}
					}
				}
			}
		}

		cv::namedWindow("mix", cv::WINDOW_NORMAL);
		cv::imshow("mix", out8);

		/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

		// ESERCIZIO 10

		std::vector<int> channels = {1, 2, 3};

		srand(unsigned(time(NULL)));
		std::random_shuffle(imgs.begin(), imgs.end());

		cv::Mat out9(image.rows, image.cols, image.type());

		for (int v = 0; v < out9.rows; ++v)
		{
			for (int u = 0; u < out9.cols; ++u)
			{
				for (int k = 0; k < out9.channels(); ++k)
				{
					out9.data[(u + v * out9.cols) * out9.elemSize() + k * out9.elemSize1()] = image.data[(u + v * image.cols) * image.elemSize() + channels[k] * image.elemSize1()];
				}
			}
		}

		cv::namedWindow("channel random", cv::WINDOW_NORMAL);
		cv::imshow("channel random", out9);

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
