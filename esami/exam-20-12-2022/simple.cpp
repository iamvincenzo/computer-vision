// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
// CANNY VINCE

void addZeroPaddingVince(const cv::Mat &src, cv::Mat &padded, const int padH, const int padW)
{
	padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(0));

	for (int v = padH; v < padded.rows - padH; ++v)
		for (int u = padW; u < padded.cols - padW; ++u)
			padded.at<u_char>(v, u) = src.at<u_char>((v - padH), (u - padW));

	return;
}

// Esercitazione 1a: convoluzione
void myfilter2DVince(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stride = 1)
{
	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPaddingVince(src, padded, padH, padW);

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
				}
			}

			out.at<int32_t>(v, u) = w_sum;
		}
	}

	return;
}

void gaussianKrnlVince(float sigma, int r, cv::Mat &krnl)
{
	krnl = cv::Mat(2 * r + 1, 1, CV_32FC1, cv::Scalar(0.0));

	float sum = 0.0;

	std::cout << std::endl;

	// calcolo kernel - formula 1D: (1/((sqrt(CV_PI * 2)*sig)) * exp(-x^2) / (2 * sig^2))))
	for (int x = -r; x <= r; ++x)
	{
		krnl.at<float>(x + r, 0) = (exp(-pow(x, 2) / (2 * pow(sigma, 2)))) / (sqrt(CV_PI * 2) * sigma);

		// calcolo della somma dei pesi
		sum += krnl.at<float>(x + r, 0);
	}

	// normalizzazione del kernel
	krnl /= sum;

	return;
}

// Esercitazione 4a: canny edge detector
void GaussianBlurVince(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride)
{
	// vertical gaussian filter creation
	cv::Mat gaussKrnl;
	gaussianKrnlVince(sigma, r, gaussKrnl);

	// horizontal gaussian filter creation
	cv::Mat gaussKrnlT;
	cv::transpose(gaussKrnl, gaussKrnlT);

	// custom convolution
	cv::Mat myfilter2DresultTmp;
	myfilter2DVince(src, gaussKrnl, myfilter2DresultTmp, stride);

	// conversion intermediate result form CV_32SC1 --> CV_8UC1
	cv::Mat conversionTmp;
	myfilter2DresultTmp.convertTo(conversionTmp, CV_8UC1);

	// custom convolution
	cv::Mat outTmp;
	myfilter2DVince(conversionTmp, gaussKrnlT, outTmp, stride);
	outTmp.convertTo(out, CV_8UC1);

	return;
}

// Esercitazione 4a: canny edge detector
void sobel3x3Vince(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
	float dataKx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float dataKy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	cv::Mat Kx(3, 3, CV_32F, dataKx);
	cv::Mat Ky(3, 3, CV_32F, dataKy);

	cv::Mat Ix;
	myfilter2DVince(src, Kx, Ix);

	cv::Mat Iy;
	myfilter2DVince(src, Ky, Iy);

	// compute magnitude
	Ix.convertTo(Ix, CV_32F);
	Iy.convertTo(Iy, CV_32F);
	cv::pow(Ix.mul(Ix) + Iy.mul(Iy), 0.5, magn);

	// compute orientation
	orient = cv::Mat(Ix.size(), CV_32FC1);
	for (int v = 0; v < Ix.rows; ++v)
	{
		for (int u = 0; u < Ix.cols; ++u)
		{
			orient.at<float>(v, u) = atan2f(Iy.at<float>(v, u), Ix.at<float>(v, u));
		}
	}

	return;
}

// Esercitazione 4a: canny edge detector
template <class T>
float bilinearVince(const cv::Mat &src, float r, float c)
{
	// r in [0,rows-1] - c in [0,cols-1]
	if (r < 0 || r > (src.rows - 1) || c < 0 || c > (src.cols - 1))
		return -1;

	// get the largest possible integer less than or equal to r/c
	int rfloor = floor(r);
	int cfloor = floor(c);
	float t = r - rfloor;
	float s = c - cfloor;

	return (src.at<T>(rfloor, cfloor)) * (1 - s) * (1 - t) +
		   (src.at<T>(rfloor, cfloor + 1)) * s * (1 - t) +
		   (src.at<T>(rfloor + 1, cfloor)) * (1 - s) * t +
		   (src.at<T>(rfloor + 1, cfloor + 1)) * t * s;
}

// Esercitazione 4a: canny edge detector
int findPeaksBilInterpInterpVince(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
	// Non Maximum Suppression

	out = cv::Mat(magn.rows, magn.cols, CV_32FC1, cv::Scalar(0.0));

	// convert orient from radiant to angles
	cv::Mat angles(orient.rows, orient.cols, orient.type(), cv::Scalar(0.0));
	orient.copyTo(angles);
	// angles *= (180 / CV_PI);

	float e1 = 0.0, e1x = 0.0, e1y = 0.0, e2 = 0.0, e2x = 0.0, e2y = 0.0;
	float theta = 0.0;

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int r = 1; r < angles.rows - 1; ++r)
	{
		for (int c = 1; c < angles.cols - 1; ++c)
		{
			theta = angles.at<float>(r, c);

			e1x = c + 1 * cos(theta);
			e1y = r + 1 * sin(theta);
			e2x = c - 1 * cos(theta);
			e2y = r - 1 * sin(theta);

			e1 = bilinearVince<float>(magn, e1y, e1x);
			e2 = bilinearVince<float>(magn, e2y, e2x);

			// magn.at<float>(r, c) is a local maxima
			if (magn.at<float>(r, c) >= e1 && magn.at<float>(r, c) >= e2)
			{
				out.at<float>(r, c) = magn.at<float>(r, c);
			}
		}
	}

	return 0;
}

// Esercitazione 4a: canny edge detector
void findAdjRecursiveVince(cv::Mat &out, const int r, const int c)
{
	// Adjacent pixel to pixel (i,j):
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	for (int i = r - 1; i <= r + 1; ++i)
	{
		for (int j = c - 1; j <= c + 1; ++j)
		{
			// se il pixel ha una valore compreso tra T-low e T-High
			if (out.at<u_char>(i, j) != 0 && out.at<u_char>(i, j) != 255)
			{
				// diventa un pixel di bordo
				out.at<u_char>(i, j) = 255;
				// analisi ricorsiva dei suoi vicini in quanto pixel di bordo
				findAdjRecursiveVince(out, i, j);
			}
		}
	}

	return;
}

// Esercitazione 4a: canny edge detector
void doubleThRecursiveVince(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
	float tmpVal = 0.0;

	out = cv::Mat(magn.rows, magn.cols, magn.type(), cv::Scalar(0.0));

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			out.at<float>(v, u) = magn.at<float>(v, u);
		}
	}

	out.convertTo(out, CV_8UC1);

	// passata 1

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			tmpVal = magn.at<float>(v, u);

			// Over T-high: keep edge
			if (tmpVal >= t2)
			{
				out.at<u_char>(v, u) = 255;
			}
			// Under T-low: remove edge
			else if (tmpVal < t1)
			{
				out.at<u_char>(v, u) = 0;
			}
		}
	}

	// passata 2

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			// per ogni pixel di bordo avvia la procedura di crescita dei suoi vicini (ricorsiva)
			if (out.at<u_char>(v, u) == 255)
			{
				findAdjRecursiveVince(out, v, u);
			}
		}
	}

	// passata 3: rimozione dei non massimi rimanenti

	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			if (out.at<u_char>(v, u) != 255)
			{
				out.at<u_char>(v, u) = 0;
			}
		}
	}

	return;
}
/////////////////////////////////////////////////////////////////////////////////

void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stridev = 1, int strideh = 1);

void gaussianKrnl(float sigma, int r, cv::Mat &krnl);

void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride = 1);

void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &ori);

template <typename T>
float bilinear(const cv::Mat &src, float r, float c);

int findPeaks(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out);

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2);

void myHoughLines(const cv::Mat &image, cv::Mat &lines, const int min_theta, const int max_theta, const int threshold);

void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img);

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
	int tlow, thigh;
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int ksize = 3;
	int stride = 1;
	float sigma = 1.0f;
	int imreadflags = cv::IMREAD_COLOR;

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

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		//////////////////////////////////
		// CANNY-PROCESSING

		cv::Mat grey;
		cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

		cv::Mat blurred, blurdisplay;
		GaussianBlur(grey, sigma, ksize / 2, blurred, stride);

		blurred.convertTo(blurdisplay, CV_8UC1);

		cv::namedWindow("Gaussian", cv::WINDOW_NORMAL);
		cv::imshow("Gaussian", blurdisplay);

		cv::Mat magnitude, orientation;
		sobel3x3(blurdisplay, magnitude, orientation);

		cv::Mat magndisplay;
		magnitude.convertTo(magndisplay, CV_8UC1);
		cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
		cv::imshow("sobel magnitude", magndisplay);

		cv::Mat ordisplay;
		orientation.copyTo(ordisplay);
		float *orp = (float *)ordisplay.data;
		for (int i = 0; i < ordisplay.cols * ordisplay.rows; ++i)
			if (magndisplay.data[i] < 50)
				orp[i] = 0;
		cv::convertScaleAbs(ordisplay, ordisplay, 255 / (2 * CV_PI));
		cv::Mat falseColorsMap;
		cv::applyColorMap(ordisplay, falseColorsMap, cv::COLORMAP_JET);
		cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
		cv::imshow("sobel orientation", falseColorsMap);

		cv::Mat nms, nmsdisplay;
		findPeaks(magnitude, orientation, nms);
		nms.convertTo(nmsdisplay, CV_8UC1);
		cv::namedWindow("edges after NMS", cv::WINDOW_NORMAL);
		cv::imshow("edges after NMS", nmsdisplay);

		cv::Mat canny;
		if (doubleTh(nms, canny, args.tlow, args.thigh))
		{
			std::cerr << "ERROR: t_low shoudl be lower than t_high" << std::endl;
			exit(1);
		}
		cv::namedWindow("Canny final result", cv::WINDOW_NORMAL);
		cv::imshow("Canny final result", canny);

		/////////////////////////////////////////////////////////////////
		// CANNY VINCE

		// gaussian smoothing
		cv::Mat smoothGrey;
		GaussianBlurVince(grey, 1, 1, smoothGrey, 1);

		// sobel filtering
		cv::Mat magn;
		cv::Mat orient;
		sobel3x3Vince(smoothGrey, magn, orient);

		cv::Mat outNms;
		findPeaksBilInterpInterpVince(magn, orient, outNms);

		float tlow = 50;
		float thigh = 200;

		cv::Mat outTHR;
		doubleThRecursiveVince(outNms, outTHR, tlow, thigh);

		// display image greyscale
		cv::namedWindow("out canny vince", cv::WINDOW_NORMAL);
		cv::imshow("out canny vince", outTHR);

		/////////////////////////////////////////////////////////////////

		//////////////////////////////////
		// HOUGH-PROCESSING VINCE

		cv::Mat blurred1;
		cv::blur(grey, blurred1, cv::Size(3, 3));

		cv::Mat contours;
		cv::Canny(blurred1, contours, 50, 200, 3);

		cv::Mat lines, lines1, lines2;
		image.copyTo(lines);
		image.copyTo(lines1);
		image.copyTo(lines2);
		myHoughLines(canny, lines, 0, 180, 150);
		myHoughLines(contours, lines1, 0, 180, 150);
		myHoughLines(outTHR, lines2, 0, 180, 150);

		// display image
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("image", image);

		// display image
		cv::namedWindow("opencv canny", cv::WINDOW_NORMAL);
		cv::imshow("opencv canny", contours);

		// display image
		cv::namedWindow("lines prof", cv::WINDOW_NORMAL);
		cv::imshow("lines prof", lines);

		// display image
		cv::namedWindow("lines opencv canny", cv::WINDOW_NORMAL);
		cv::imshow("lines opencv canny", lines1);

		// display image
		cv::namedWindow("lines opencv vince", cv::WINDOW_NORMAL);
		cv::imshow("lines opencv vince", lines2);

		//////////////////////////////////

		//////////////////////////////////
		// HOUGH-PROCESSING OPENCV

		cv::Mat lines3, lines4, lines5;
		image.copyTo(lines3);
		image.copyTo(lines4);
		image.copyTo(lines5);

		// myHoughLines(canny, lines3, 0, 180, 150);
		//  Standard Hough Line Transform
		std::vector<cv::Vec2f> liness;						  // will hold the results of the detection
		HoughLines(canny, liness, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < liness.size(); i++)
		{
			float rho = liness[i][0], theta = liness[i][1];
			cv::Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(lines3, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
		}

		// myHoughLines(contours, lines4, 0, 180, 150);
		liness.clear();											 // will hold the results of the detection
		HoughLines(contours, liness, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < liness.size(); i++)
		{
			float rho = liness[i][0], theta = liness[i][1];
			cv::Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(lines4, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
		}

		// myHoughLines(outTHR, lines5, 0, 180, 150);
		liness.clear();										   // will hold the results of the detection
		HoughLines(outTHR, liness, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < liness.size(); i++)
		{
			float rho = liness[i][0], theta = liness[i][1];
			cv::Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(lines5, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
		}

		// display image
		cv::namedWindow("lines prof 3", cv::WINDOW_NORMAL);
		cv::imshow("lines prof 3", lines3);

		// display image
		cv::namedWindow("lines opencv canny 4", cv::WINDOW_NORMAL);
		cv::imshow("lines opencv canny 4", lines4);

		// display image
		cv::namedWindow("lines opencv vince 5", cv::WINDOW_NORMAL);
		cv::imshow("lines opencv vince 5", lines5);

		//////////////////////////////////

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
		case 'G':
			sigma *= 2;
			std::cout << "Sigma: " << sigma << std::endl;
			break;
		case 'g':
			sigma /= 2;
			std::cout << "Sigma: " << sigma << std::endl;
			break;
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
		case 'q':
			exit(0);
			break;
		}

		frame_number++;
	}

	return 0;
}

#include <unistd.h>
bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
	int c;
	args.wait_t = 0;
	args.tlow = 50;
	args.thigh = 200;

	while ((c = getopt(argc, argv, "hi:t:L:H:")) != -1)
		switch (c)
		{
		case 'H':
			args.thigh = atoi(optarg);
			break;
		case 'L':
			args.tlow = atoi(optarg);
			break;
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
					  << "   -H arg                   Canny high threshold                            " << std::endl
					  << "   -L arg                   Canny low  threshold                            " << std::endl
					  << "   -t arg                   wait before next frame (ms)                     " << std::endl
					  << std::endl;
			return false;
		}
	return true;
}

void addPadding(const cv::Mat image, cv::Mat &out, int vPadding, int hPadding)
{
	out = cv::Mat(image.rows + vPadding * 2, image.cols + hPadding * 2, image.type(), cv::Scalar(0));

	for (int row = vPadding; row < out.rows - vPadding; ++row)
	{
		for (int col = hPadding; col < out.cols - hPadding; ++col)
		{
			for (int k = 0; k < out.channels(); ++k)
			{
				out.data[((row * out.cols + col) * out.elemSize() + k * out.elemSize1())] = image.data[(((row - vPadding) * image.cols + col - hPadding) * image.elemSize() + k * image.elemSize1())];
			}
		}
	}

#if DEBUG
	std::cout << "Padded image " << out.rows << "x" << out.cols << std::endl;
	cv::namedWindow("Padded", cv::WINDOW_NORMAL);
	cv::imshow("Padded", out);
	unsigned char key = cv::waitKey(0);
#endif
}

/*
 Src: singolo canale uint8
 Krn: singolo canale float32 di dimensioni dispari
 Out: singolo canale int32
 Stride: intero (default 1)
*/
void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stridev, int strideh)
{

	if (!src.rows % 2 || !src.cols % 2)
	{
		std::cerr << "myfilter2D(): ERROR krn has not odd size!" << std::endl;
		exit(1);
	}

	int outsizey = (src.rows + (krn.rows / 2) * 2 - krn.rows) / (float)stridev + 1;
	int outsizex = (src.cols + (krn.cols / 2) * 2 - krn.cols) / (float)strideh + 1;
	out = cv::Mat(outsizey, outsizex, CV_32SC1);
	// std::cout << "Output image " << out.rows << "x" << out.cols << std::endl;

	cv::Mat image;
	addPadding(src, image, krn.rows / 2, krn.cols / 2);

	int xc = krn.cols / 2;
	int yc = krn.rows / 2;

	int *outbuffer = (int *)out.data;
	float *kernel = (float *)krn.data;

	for (int i = 0; i < out.rows; ++i)
	{
		for (int j = 0; j < out.cols; ++j)
		{
			int origy = i * stridev + yc;
			int origx = j * strideh + xc;
			float sum = 0;
			for (int ki = -yc; ki <= yc; ++ki)
			{
				for (int kj = -xc; kj <= xc; ++kj)
				{
					sum += image.data[(origy + ki) * image.cols + (origx + kj)] * kernel[(ki + yc) * krn.cols + (kj + xc)];
				}
			}
			outbuffer[i * out.cols + j] = sum;
		}
	}
}

void gaussianKrnl(float sigma, int r, cv::Mat &krnl)
{
	float kernelSum = 0;
	krnl = cv::Mat(r * 2 + 1, 1, CV_32FC1);

	int yc = krnl.rows / 2;

	float sigma2 = pow(sigma, 2);

	for (int i = 0; i <= yc; i++)
	{
		int y2 = pow(i - yc, 2);
		float gaussValue = pow(M_E, -(y2) / (2 * sigma2));

		kernelSum += gaussValue;

		if (i != yc)
		{
			kernelSum += gaussValue;
		}

		((float *)krnl.data)[i] = gaussValue;
		((float *)krnl.data)[krnl.rows - i - 1] = gaussValue;
	}

	// Normalize.
	for (int i = 0; i < krnl.rows; i++)
	{
		((float *)krnl.data)[i] /= kernelSum;
	}
}

#define SEPARABLE
void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride)
{
	cv::Mat vg, hg;

	gaussianKrnl(sigma, r, vg);

#ifdef SEPARABLE
	hg = vg.t();
	std::cout << "DEBUG: Horizontal Gaussian Kernel:\n"
			  << hg << "\nSum: " << cv::sum(hg)[0] << std::endl;
	cv::Mat tmp;
	myfilter2D(src, hg, tmp, 1, stride);
	tmp.convertTo(tmp, CV_8UC1);
	myfilter2D(tmp, vg, out, stride, 1);
#else
	myfilter2D(src, vg * vg.t(), out, stride);
	std::cout << "DEBUG: Square Gaussian Kernel:\n"
			  << vg * vg.t() << "\nSum: " << cv::sum(vg * vg.t())[0] << std::endl;
#endif
}

// src uint8
// magn float32
// or float32
void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &ori)
{
	// SOBEL FILTERING
	// void cv::Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
	// sobel verticale come trasposto dell'orizzontale

	cv::Mat ix, iy;
	cv::Mat h_sobel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1);

	cv::Mat v_sobel = h_sobel.t();

	myfilter2D(src, h_sobel, ix, 1, 1);
	myfilter2D(src, v_sobel, iy, 1, 1);
	ix.convertTo(ix, CV_32FC1);
	iy.convertTo(iy, CV_32FC1);

	// compute magnitude
	cv::pow(ix.mul(ix) + iy.mul(iy), 0.5, magn);
	// compute orientation
	ori = cv::Mat(src.size(), CV_32FC1);
	float *dest = (float *)ori.data;
	float *srcx = (float *)ix.data;
	float *srcy = (float *)iy.data;

	for (int i = 0; i < ix.rows * ix.cols; ++i)
		dest[i] = atan2f(srcy[i], srcx[i]) + 2 * CV_PI;
}

template <typename T>
float bilinear(const cv::Mat &src, float r, float c)
{

	float yDist = r - int(r);
	float xDist = c - int(c);

	int value =
		src.at<T>(r, c) * (1 - yDist) * (1 - xDist) +
		src.at<T>(r + 1, c) * (yDist) * (1 - xDist) +
		src.at<T>(r, c + 1) * (1 - yDist) * (xDist) +
		src.at<T>(r + 1, c + 1) * yDist * xDist;

	return value;
}

int findPeaks(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{

	out = cv::Mat(magn.size(), magn.type(), cv::Scalar(0));
	for (int r = 1; r < magn.rows - 1; r++)
	{
		for (int c = 1; c < magn.cols - 1; c++)
		{

			float theta = orient.at<float>(r, c);
			float e1x = c + cos(theta);
			float e1y = r + sin(theta);
			float e2x = c - cos(theta);
			float e2y = r - sin(theta);

			float e1 = bilinear<float>(magn, e1y, e1x);
			float e2 = bilinear<float>(magn, e2y, e2x);
			float p = magn.at<float>(r, c);

			if (p < e1 || p < e2)
			{
				p = 0;
			}

			out.at<float>(r, c) = p;
		}
	}
	return 0;
}

/*
 magnitude: singolo canale float32
 out: singolo canale uint8 binarizzato
 t1 e t2: soglie
*/
int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
	cv::Mat first = cv::Mat(magn.size(), CV_8UC1);
	float p; // little optimization (complier should cope with this)
	if (t1 >= t2)
		return 1;

	int tm = t1 + (t2 - t1) / 2;

	std::vector<cv::Point2i> strong;
	std::vector<cv::Point2i> low;
	for (int r = 0; r < magn.rows; r++)
	{
		for (int c = 0; c < magn.cols; c++)
		{
			if ((p = magn.at<float>(r, c)) >= t2)
			{
				first.at<uint8_t>(r, c) = 255;
				strong.push_back(cv::Point2i(c, r)); // BEWARE at<>() and point2i() use a different coords order...
			}
			else if (p <= t1)
			{
				first.at<uint8_t>(r, c) = 0;
			}
			else
			{
				first.at<uint8_t>(r, c) = tm;
				low.push_back(cv::Point2i(c, r));
			}
		}
	}

	cv::namedWindow("Intermediate Canny result", cv::WINDOW_NORMAL);
	cv::imshow("Intermediate Canny result", first);
	first.copyTo(out);

	// grow points > t2
	while (!strong.empty())
	{
		cv::Point2i p = strong.back();
		strong.pop_back();
		// std::cout << p.y << " " << p.x << std::endl;
		for (int ox = -1; ox <= 1; ++ox)
			for (int oy = -1; oy <= 1; ++oy)
			{
				int nx = p.x + ox;
				int ny = p.y + oy;
				if (nx > 0 && nx < out.cols && ny > 0 && ny < out.rows && out.at<uint8_t>(ny, nx) == tm)
				{
					// std::cerr << ".";
					out.at<uint8_t>(ny, nx) = 255;
					strong.push_back(cv::Point2i(nx, ny));
				}
			}
	}

	// wipe out residual pixels < t2
	while (!low.empty())
	{
		cv::Point2i p = low.back();
		low.pop_back();
		if (out.at<uint8_t>(p.y, p.x) < 255)
			out.at<uint8_t>(p.y, p.x) = 0;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// HOUGH VINCE

// Trasformata di Hough: linee
void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img)
{
	if (theta >= 45 && theta <= 135)
	{
		// y = (r - x cos(t)) / sin(t)
		p1.x = 0;
		p1.y = ((double)(rho - (dist / 2)) - ((p1.x - (img.cols / 2)) * cos(theta * CV_PI / 180))) / sin(theta * CV_PI / 180) + (img.rows / 2);
		p2.x = img.cols;
		p2.y = ((double)(rho - (dist / 2)) - ((p2.x - (img.cols / 2)) * cos(theta * CV_PI / 180))) / sin(theta * CV_PI / 180) + (img.rows / 2);
	}
	else
	{
		// x = (r - y sin(t)) / cos(t);
		p1.y = 0;
		p1.x = ((double)(rho - (dist / 2)) - ((p1.y - (img.rows / 2)) * sin(theta * CV_PI / 180))) / cos(theta * CV_PI / 180) + (img.cols / 2);
		p2.y = img.rows;
		p2.x = ((double)(rho - (dist / 2)) - ((p2.y - (img.rows / 2)) * sin(theta * CV_PI / 180))) / cos(theta * CV_PI / 180) + (img.cols / 2);
	}

	return;
}

// Trasformata di Hough: linee
void myHoughLines(const cv::Mat &image, cv::Mat &lines, const int min_theta, const int max_theta, const int threshold)
{
	if (image.type() != CV_8UC1)
	{
		std::cerr << "houghLines() - ERROR: the image is not uint8." << std::endl;
		exit(1);
	}

	if (min_theta < 0 || min_theta >= max_theta)
	{
		std::cerr << "houghLines() - ERROR: the minimum value of theta min_theta is out of the valid range [0, max_theta)." << std::endl;
		exit(1);
	}

	if (max_theta <= min_theta || max_theta > 180)
	{
		std::cerr << "houghLines() - ERROR: the maximum value of theta max_theta is out of the valid range (min_theta, PI]." << std::endl;
		exit(1);
	}

	int max;

	if (image.rows > image.cols)
		max = image.rows;
	else
		max = image.cols;

	int max_distance = pow(2, 0.5) * max / 2;

	std::vector<int> acc_row(max_theta - min_theta + 1, 0);

	std::vector<std::vector<int>> accumulator(2 * max_distance, acc_row);

	for (int r = 0; r < image.rows; ++r)
	{
		for (int c = 0; c < image.cols; ++c)
		{
			// check if the current pixel is an edge pixel
			if (image.at<u_char>(r, c) > 0)
			{
				int rho;
				// loop over all possible values of theta
				for (int theta = min_theta; theta <= max_theta; ++theta)
				{
					rho = (c - image.cols / 2) * cos(theta * CV_PI / 180) + (r - image.rows / 2) * sin(theta * CV_PI / 180); // compute the value of rho

					++accumulator[rho + max_distance][theta]; // increase the (rho, theta) position in the accumulator
				}
			}
		}
	}

	cv::Mat acc(2 * max_distance, max_theta - min_theta, CV_8UC1);

	for (int r = 0; r < 2 * max_distance; ++r)
	{
		for (int t = min_theta; t <= max_theta; ++t)
		{
			acc.at<u_char>(r, t) = accumulator[r][t];
		}
	}

	cv::namedWindow("Accumulator", cv::WINDOW_NORMAL);
	cv::imshow("Accumulator", acc);

	cv::Point start_point, end_point;

	for (int r = 0; r < acc.rows; ++r)
	{
		for (int t = min_theta; t < acc.cols; ++t)
		{
			if (accumulator[r][t] >= threshold)
			{
				// convert to cartesian coordinates
				myPolarToCartesian(r, t, start_point, end_point, acc.rows, image);

				// draw a red line
				cv::line(lines, start_point, end_point, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
			}
		}
	}

	return;
}
