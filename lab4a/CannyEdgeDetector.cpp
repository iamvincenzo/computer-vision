// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>
#include <iomanip>

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

bool checkOdd(const cv::Mat &krnl)
{
	if (krnl.rows % 2 == 0 || krnl.cols % 2 == 0)
		return false;
	else
		return true;
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

	// display padded image
	cv::namedWindow("padded image", cv::WINDOW_NORMAL);
	cv::imshow("padded image", padded);

	return;
}

void myfilter2D(const cv::Mat &src, const cv::Mat &krnl, cv::Mat &out, int stride = 1)
{
	if (!checkOdd(krnl))
	{
		std::cout << "Error: the kernel must be odd!" << std::endl;
		return;
	}

	int padH = (krnl.rows - 1) / 2;
	int padW = (krnl.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	// output_height = (int) ((input height + padding height top + padding height bottom - kernel height) / (stride height) + 1)
	// output_width = (int) ((input width + padding width right + padding width left - kernel width) / (stride width) + 1)
	out = cv::Mat((int)((src.rows + 2 * padH - krnl.rows) / stride + 1), (int)((src.cols + 2 * padW - krnl.cols) / stride + 1), CV_32SC1);

	float g_kl;
	float w_sum;

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			w_sum = 0.0;

			for (int k = 0; k < krnl.rows; ++k)
			{
				for (int l = 0; l < krnl.cols; ++l)
				{
					g_kl = krnl.at<float>(k, l);
					w_sum += g_kl * (float)padded.at<u_char>((v * stride) + k, (u * stride) + l);
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

void gaussianKrnl2D(float sigma, int r, cv::Mat &krnl)
{
	krnl = cv::Mat(2 * r + 1, 2 * r + 1, CV_32FC1, cv::Scalar(0.0));

	float t = 0.0, sum = 0.0, s = 2 * sigma * sigma;

	// calcolo kernel
	for (int x = -r; x <= r; ++x)
	{
		for (int y = -r; y <= r; ++y)
		{
			t = sqrt((x * x) + (y * y));

			krnl.at<float>(x + r, y + r) = (exp(-(t * t) / s)) / (CV_PI * s);

			// calcolo della somma dei pesi
			sum += krnl.at<float>(x + r, y + r);
		}
	}

	// normalizzazione del kernel
	for (int i = 0; i < krnl.rows; ++i)
	{
		for (int j = 0; j < krnl.cols; ++j)
		{
			krnl.at<float>(i, j) /= sum;
		}
	}

	// krnl /= sum;

	// stampa kernel
	std::cout << "Gaussian Kernel 2D:\n"
			  << std::endl;

	// // fissa la precisione a 4 cifre decimali
	// std::cout << std::fixed;
	// std::cout << std::setprecision(4);

	for (int v = 0; v < krnl.rows; ++v)
	{
		for (int u = 0; u < krnl.cols; ++u)
		{
			std::cout << krnl.at<float>(v, u) << "\t";
		}

		std::cout << std::endl;
	}

	// display kernel
	cv::namedWindow("gaussian krnl 2D", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 2D", krnl);

	return;
}

void gaussianKrnl(float sigma, int r, cv::Mat &krnl)
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

	// stampa kernel
	std::cout << "Vertical Gaussian Kernel - 1D:\n"
			  << std::endl;

	for (int v = 0; v < krnl.rows; ++v)
	{
		for (int u = 0; u < krnl.cols; ++u)
		{
			std::cout << krnl.at<float>(v, u) << "\t";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;

	// display kernel
	cv::namedWindow("gaussian krnl 1D - vertical", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 1D - vertical", krnl);

	return;
}

void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride)
{
	// vertical gaussian filter creation
	cv::Mat gaussKrnl;
	gaussianKrnl(sigma, r, gaussKrnl);

	// horizontal gaussian filter creation
	cv::Mat gaussKrnlT;
	cv::transpose(gaussKrnl, gaussKrnlT);

	// display horizontal kernel
	cv::namedWindow("gaussian krnl 1D - horizontal", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 1D - horizontal", gaussKrnlT);

	// custom convolution
	cv::Mat myfilter2DresultTmp;
	myfilter2D(src, gaussKrnl, myfilter2DresultTmp, stride);

	// conversion intermediate result form CV_32SC1 --> CV_8UC1
	cv::Mat conversionTmp;
	myfilter2DresultTmp.convertTo(conversionTmp, CV_8UC1);

	// custom convolution
	cv::Mat outTmp;
	myfilter2D(conversionTmp, gaussKrnlT, outTmp, stride);
	outTmp.convertTo(out, CV_8UC1);

	return;
}

void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
	float dataKx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float dataKy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	cv::Mat Kx(3, 3, CV_32F, dataKx);
	cv::Mat Ky(3, 3, CV_32F, dataKy);

	cv::Mat Ix;
	myfilter2D(src, Kx, Ix);

	cv::Mat Iy;
	myfilter2D(src, Ky, Iy);

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

	// scale on 0-255 range
	// per quanto riguarda Sobel, si sfrutta convertScaleAbs e non convertTo() perchè ci sono i valori negativi
	cv::Mat aIx, aIy, amagn;
	cv::convertScaleAbs(Ix, aIx);
	cv::convertScaleAbs(Iy, aIy);
	cv::convertScaleAbs(magn, amagn);

	// display vertical sobel
	cv::namedWindow("vertical sobel", cv::WINDOW_NORMAL);
	cv::imshow("vertical sobel", aIx);

	// display vertical sobel
	cv::namedWindow("horizontal sobel", cv::WINDOW_NORMAL);
	cv::imshow("horizontal sobel", aIy);

	// display sobel magnitude
	cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude", amagn);

	// trick to display orientation
	cv::Mat adjMap;
	cv::convertScaleAbs(orient, adjMap, 255 / (2 * CV_PI));
	cv::Mat falseColorsMap;
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN); // COLORMAP_JET
	cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
	cv::imshow("sobel orientation", falseColorsMap);

	return;
}

template <class T>
float bilinear(const cv::Mat &src, float r, float c)
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

int findPeaks3x3(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
	// Non Maximum Suppression
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	out = cv::Mat(magn.rows, magn.cols, magn.type(), cv::Scalar(0.0));

	float e1 = 255.0, e2 = 255.0, theta = 0.0;

	// convert orient from radiant to angles
	cv::Mat angles(orient.rows, orient.cols, orient.type(), cv::Scalar(0.0));
	orient.copyTo(angles);
	angles *= (180 / CV_PI);

	for (int v = 0; v < angles.rows; ++v)
	{
		for (int u = 0; u < angles.cols; ++u)
		{
			// if (angles.at<float>(v, u) < 0)
			// {
			// 	angles.at<float>(v, u) += 180;
			// }

			// so that there aren't negative angles
			// all angles in range 180 (CV_PI = 180° is the atan2 periodicity)
			while (angles.at<float>(v, u) < 0)
			{
				angles.at<float>(v, u) += 180;
			}

			// all angles in range 180 (CV_PI = 180° is the atan2 periodicity)
			while (angles.at<float>(v, u) > 180)
			{
				angles.at<float>(v, u) -= 180;
			}
		}
	}

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < angles.rows - 1; ++v)
	{
		for (int u = 1; u < angles.cols - 1; ++u)
		{
			theta = angles.at<float>(v, u);

			// angle 0
			if ((0 <= theta && theta < 22.5) || (157.5 <= theta && theta <= 180))
			{
				e1 = magn.at<float>(v, u + 1);
				e2 = magn.at<float>(v, u - 1);
			}
			// angle 45
			else if (22.5 <= theta && theta < 67.5)
			{
				// gradient oblique direction
				e1 = magn.at<float>(v + 1, u - 1);
				e2 = magn.at<float>(v - 1, u + 1);
			}
			// angle 90
			else if (67.5 <= theta && theta < 112.5)
			{
				// gradient vertical direction
				e1 = magn.at<float>(v + 1, u);
				e2 = magn.at<float>(v - 1, u);
			}
			// angle 135
			else if (112.5 <= theta && theta < 157.5)
			{
				// gradient oblique direction
				e1 = magn.at<float>(v - 1, u - 1);
				e2 = magn.at<float>(v + 1, u + 1);
			}

			// magn.at<float>(r, c) is a local maxima
			if (magn.at<float>(v, u) >= e1 && magn.at<float>(v, u) >= e2)
			{
				out.at<float>(v, u) = magn.at<float>(v, u);
			}
		}
	}

	// scale on 0-255 range
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	// in realtà, è possibile usare convertTo() perchè nella magnitude non ci sono valori negativi (per come è definita non possono esserci)
	out.convertTo(outDisplay, CV_8UC1);
	// display sobel magnitude
	cv::namedWindow("sobel magnitude NMS - 3x3mask", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude NMS - 3x3mask", outDisplay);

	return 0;
}

int findPeaksBilInterpInterp(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
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

			e1 = bilinear<float>(magn, e1y, e1x);
			e2 = bilinear<float>(magn, e2y, e2x);

			// magn.at<float>(r, c) is a local maxima
			if (magn.at<float>(r, c) >= e1 && magn.at<float>(r, c) >= e2)
			{
				out.at<float>(r, c) = magn.at<float>(r, c);
			}
		}
	}

	// scale on 0-255 range
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);

	// display sobel magnitude
	cv::namedWindow("sobel magnitude NMS - bilinInterp", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude NMS - bilinInterp", outDisplay);

	return 0;
}

void findOptTreshs(const cv::Mat &src, float &tlow, float &thigh)
{
	float sum = 0.0;
	int N = 0;
	float medianPix = 0.0;

	for (int v = 0; v < src.rows; ++v)
	{
		for (int u = 0; u < src.cols; ++u)
		{
			sum += (float)src.at<u_char>(v, u);
			++N;
		}
	}

	medianPix = sum / N;

	// max(0, 0.7 * medianPix)
	if (0 > 0.7 * medianPix)
		tlow = 0;
	else
		tlow = 0.7 * medianPix;

	// min(255, 1.3 * medianPix)
	if (255 < 1.3 * medianPix)
		thigh = 255;
	else
		thigh = 1.3 * medianPix;

	std::cout << "\n(doubleTh) Optiaml tresholds: \ntlow: " << tlow << " - thigh: " << thigh << std::endl;

	return;
}

void findAdj(const cv::Mat &magn, cv::Mat &out, const int r, const int c, const float tlow)
{
	// Adjacent pixel to pixel (i,j):
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	bool ignore = false;

	for (int i = r - 1; i <= r + 1; ++i)
	{
		for (int j = c - 1; j <= c + 1; ++j)
		{
			if (i == r && j == c)
			{
				ignore = true;
				continue;
			}

			if (magn.at<float>(i, j) >= tlow)
			{
				out.at<u_char>(i, j) = 255;
			}
		}

		if (ignore)
			continue;
	}

	return;
}

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
	float tmpVal = 0.0;

	out = cv::Mat(magn.rows, magn.cols, CV_8UC1, cv::Scalar(0));

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
		}
	}

	// passata 2

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			if (out.at<u_char>(v, u) == 255)
			{
				findAdj(magn, out, v, u, t1);
			}
		}
	}

	// display image greyscale
	cv::namedWindow("out hysteresis - doubleTh", cv::WINDOW_NORMAL);
	cv::imshow("out hysteresis - doubleTh", out);

	return 0;
}

void findAdjRecursive(cv::Mat &out, const int r, const int c)
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
				findAdjRecursive(out, i, j);
			}
		}
	}

	return;
}

void doubleThRecursive(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
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
				findAdjRecursive(out, v, u);
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

	// display image greyscale
	cv::namedWindow("out hysteresis - doubleThRecursive", cv::WINDOW_NORMAL);
	cv::imshow("out hysteresis - doubleThRecursive", out);

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

		std::cout << "The image has " << image.channels()
				  << " channels, the size is " << image.rows
				  << "x" << image.cols << " pixels "
				  << " the type is " << image.type()
				  << " the pixel size is " << image.elemSize()
				  << " and each channel is " << image.elemSize1()
				  << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl
				  << std::endl;

		//////////////////////
		// processing code here

		cv::Mat grey;
		cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY); // myFilter2Dersione dell'immagine da RGB a scala di grigi e salva il risultato in grey

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// display image greyscale
		cv::namedWindow("grey", cv::WINDOW_NORMAL);
		cv::imshow("grey", grey);

		// gaussian smoothing
		cv::Mat smoothGrey;
		GaussianBlur(grey, 1, 1, smoothGrey, 1);

		// sobel filtering
		cv::Mat magn;
		cv::Mat orient;
		sobel3x3(smoothGrey, magn, orient);

		// cv::Mat outNms1;
		// findPeaks3x3(magn, orient, outNms1); --> alternativa con maschera 3x3 piuttosto che con interpolazione bilineare
		cv::Mat outNms;
		findPeaksBilInterpInterp(magn, orient, outNms);

		float tlow;
		float thigh;
		findOptTreshs(smoothGrey, tlow, thigh);

		// cv::Mat outTH;
		// doubleTh(outNms, outTH, tlow, thigh); // alternativa al metodo ricorsivo (non sicuro della correttezza)

		tlow = 50;
		thigh = 100;

		cv::Mat outTHR;
		doubleThRecursive(outNms, outTHR, tlow, thigh);

		/////////////////////

		// wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << std::endl
				  << "key " << int(key) << std::endl;

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

/* old wrong stuff

bool findAdj(const cv::Mat &magn, const int r, const int c, const int thigh)
{
	// Adjacent pixel to pixel (i,j):
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	bool ignore = false;

	for (int i = r - 1; i <= r + 1; ++i)
	{
		for (int j = c - 1; j <= c + 1; ++j)
		{
			if (i == r && j == c)
			{
				ignore = true;
				continue;
			}

			if (magn.at<float>(i, j) >= thigh)
			{
				return true;
			}
		}

		if (ignore)
			continue;
	}

	return false;
}

int hist(const cv::Mat &magn, cv::Mat &out, float lowThrRatio = 0.05, float highThRatio = 0.08) // float highThRatio = 0.09
{
	// altra versione che implementa in maniera differente i metodi:
	// findOptTreshs + doubleTh

	float max = 0.0;

	for (int v = 0; v < magn.rows; ++v)
	{
		for (int u = 0; u < magn.cols; ++u)
		{
			if (magn.at<float>(v, u) > max)
			{
				max = magn.at<float>(v, u);
			}
		}
	}

	float t2 = max * highThRatio;
	float t1 = t2 * lowThrRatio;

	std::cout << "\n(hist) Optiaml tresholds: \ntlow: " << t1 << " - thigh: " << t2 << std::endl;

	float tmpVal = 0.0;

	out = cv::Mat(magn.rows, magn.cols, CV_8UC1, cv::Scalar(0));

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
			// Between T-low and T-high: keep if adjacent edges above T-low
			else if (tmpVal >= t1 && tmpVal <= t2)
			{
				// there is an adiacent edge above T-low
				if (findAdj(magn, v, u, t2))
				{
					out.at<u_char>(v, u) = 255;
				}
			}
		}
	}

	// display image greyscale
	cv::namedWindow("out hysteresis - hist", cv::WINDOW_NORMAL);
	cv::imshow("out hysteresis - hist", out);

	return 0;
}
*/