// OpenCV
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
	int k;
	int threshold;
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

/*
  Computes the foreground of the current frame.

  Parameters:
	- prevKFrames   -> previous k frames
	- currI         -> current frame
	- out           -> output (foreground)
	- th            -> threshold
*/
void computeForeground(const std::vector<cv::Mat> prevKFrames, const cv::Mat &currI, cv::Mat &out, int th)
{
	int k = prevKFrames.size();
	int sum, avg, diff;

	// per ogni pixel (di tutte le immagini nel vettore)
	for (int i = 0; i < currI.rows * currI.cols * currI.elemSize(); ++i)
	{
		sum = 0;

		// di ogni immagine
		for (int j = 0; j < prevKFrames.size(); ++j)
		{
			sum += prevKFrames[j].data[i]; // si sommano i valori dei pixel
		}

		avg = sum / k; // si calcola la media della somma del pixel i-esimo di ogni immagine j-esima

		diff = abs(currI.data[i] - avg); // si calcola la differenza |I(i,j) - B(i,j)|

		if (diff > th)
			out.data[i] = 255;
		else
			out.data[i] = 0;
	}
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_COLOR;
	std::vector<cv::Mat> lastKFrames;
	cv::Mat foreground;

	std::cout << "Lab 02 - Background subraction" << std::endl;
	std::cout << "Method: running average" << std::endl;

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

		// creazione del foreground
		foreground = cv::Mat(image.rows, image.cols, image.type());

		// se all'interno del vettore sono presenti almeno k immagini
		if (lastKFrames.size() == args.k)
			computeForeground(lastKFrames, image, foreground, args.threshold);

		// inserisci l'elemento all'interno del vettore (l'elemento viene posto alla fine)
		lastKFrames.push_back(image);

		// se all'interno del vettore ci sono più di k elementi, si rimuove il primo elemento (quello più vecchio)
		if (lastKFrames.size() > args.k)
			lastKFrames.erase(lastKFrames.begin());

		// display image
		cv::namedWindow("Original image", cv::WINDOW_NORMAL);
		cv::imshow("Original image", image);

		cv::namedWindow("Foreground", cv::WINDOW_NORMAL);
		cv::imshow("Foreground", foreground);

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
		case 't':
			args.threshold -= 5;
			std::cout << "New threshold value: " << args.threshold << std::endl;
			break;
		case 'T':
			args.threshold += 5;
			std::cout << "New threshold value: " << args.threshold << std::endl;
			break;
		case 'k':
			if (args.k > 1)
			{
				args.k--;
				std::cout << "New k value: " << args.k << std::endl;
			}
			break;
		case 'K':
			args.k++;
			std::cout << "New k value: " << args.k << std::endl;
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
	args.k = 5;
	args.threshold = 40;

	while ((c = getopt(argc, argv, "hi:t:k:f:")) != -1)
		switch (c)
		{
		case 't':
			args.wait_t = atoi(optarg);
			break;
		case 'i':
			args.image_name = optarg;
			break;
		case 'k':
			args.k = atoi(optarg);
			break;
		case 'f':
			args.threshold = atoi(optarg);
			break;
		case 'h':
		default:
			std::cout << "Usage: " << argv[0] << " -i <image_name>" << std::endl;
			std::cout << "To exit:  type q" << std::endl
					  << std::endl;
			std::cout << "To increment threshold value:  type T" << std::endl;
			std::cout << "To decrement threshold value:  type t" << std::endl;
			std::cout << "To increment k value:  type K" << std::endl;
			std::cout << "To decrement k value:  type k" << std::endl
					  << std::endl;
			std::cout << "Allowed options:" << std::endl
					  << "   -h                       produce help message" << std::endl
					  << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
					  << "   -k arg                   number of considered previous frame [default = 5]" << std::endl
					  << "   -f arg                   threshold value [default = 40]" << std::endl
					  << "   -t arg                   wait before next frame (ms)" << std::endl
					  << std::endl;
			return false;
		}
	return true;
}

#endif
