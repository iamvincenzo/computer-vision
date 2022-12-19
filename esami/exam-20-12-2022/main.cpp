// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

struct ArgumentList
{
    std::string image_name; //!< image file name
    int wait_t;             //!< waiting time
    int tlow, thigh;
};

/* prototipi di funzioni */

bool ParseInputs(ArgumentList &args, int argc, char **argv);

// Canny

bool checkOdd(const cv::Mat &krnl);

void addZeroPadding(const cv::Mat &src, cv::Mat &padded, const int padH, const int padW);

void myfilter2D(const cv::Mat &src, const cv::Mat &krnl, cv::Mat &out, int stride);

void gaussianKrnl2D(float sigma, int r, cv::Mat &krnl);

void gaussianKrnl(float sigma, int r, cv::Mat &krnl);

void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride);

void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient);

template <class T>
float bilinear(const cv::Mat &src, float r, float c);

void findPeaks3x3(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out);

void findPeaksBilInterpInterp(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out);

void findOptTreshs(const cv::Mat &src, float &tlow, float &thigh);

void findAdjRecursive(cv::Mat &out, const int r, const int c);

void doubleThRecursive(const cv::Mat &magn, cv::Mat &out, float t1, float t2);

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2);

// Hough

void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img);

void myHoughTransfLines(const cv::Mat &image, cv::Mat &lines, const int minTheta, const int maxTheta, const int threshold);

/* funzione principale */

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

        //////////////////////////////////
        // CANNY-PROCESSING

        cv::Mat grey;
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

        cv::namedWindow("image", cv::WINDOW_NORMAL);
        cv::imshow("image", image);

        cv::namedWindow("grey", cv::WINDOW_NORMAL);
        cv::imshow("grey", grey);

        cv::Mat smoothGrey;
        GaussianBlur(grey, 1, 1, smoothGrey, 1);

        cv::namedWindow("gaussian blur", cv::WINDOW_NORMAL);
        cv::imshow("gaussian blur", smoothGrey);

        cv::Mat magn, orient;
        sobel3x3(smoothGrey, magn, orient);

        cv::Mat magnDisplay;
        cv::convertScaleAbs(magn, magnDisplay);
        cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
        cv::imshow("sobel magnitude", magnDisplay);

        cv::Mat orientDisplay;
        orient.copyTo(orientDisplay);
        float *orp = (float *)orientDisplay.data;
        for (int i = 0; i < orientDisplay.cols * orientDisplay.rows; ++i)
            if (magnDisplay.data[i] < 50)
                orp[i] = 0;
        cv::convertScaleAbs(orientDisplay, orientDisplay, 255 / (2 * CV_PI));
        cv::Mat falseColorsMap;
        cv::applyColorMap(orientDisplay, falseColorsMap, cv::COLORMAP_JET);
        cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
        cv::imshow("sobel orientation", falseColorsMap);

        cv::Mat outNms, OutNmsDisplay;
        // findPeaksBilInterpInterp(magn, orient, outNms);
        findPeaks3x3(magn, orient, outNms);

        outNms.convertTo(OutNmsDisplay, CV_8UC1);
        cv::namedWindow("non-maxima suppression", cv::WINDOW_NORMAL);
        cv::imshow("non-maxima suppression", OutNmsDisplay);

        float tlow, thigh;
        findOptTreshs(smoothGrey, tlow, thigh);

        // final step canny: vince

        cv::Mat outThr;
        doubleThRecursive(outNms, outThr, tlow, thigh);

        cv::namedWindow("vince canny", cv::WINDOW_NORMAL);
        cv::imshow("vince canny", outThr);

        // final step canny: prof

        cv::Mat canny;
        doubleTh(outNms, canny, tlow, thigh);

        cv::namedWindow("prof canny", cv::WINDOW_NORMAL);
        cv::imshow("prof canny", canny);

        // final step canny: opencv

        cv::Mat blurred1;
        cv::blur(grey, blurred1, cv::Size(3, 3));

        cv::Mat contours;
        cv::Canny(blurred1, contours, tlow, thigh, 3);

        cv::namedWindow("opencv canny", cv::WINDOW_NORMAL);
        cv::imshow("opencv canny", contours);

        //////////////////////////////////
        // HOUGH-PROCESSING

        int houthTh = thigh;

        cv::Mat lines, lines1, lines2;
        image.copyTo(lines);
        image.copyTo(lines1);
        image.copyTo(lines2);

        myHoughTransfLines(outThr, lines, 0, 180, houthTh);    // vince
        myHoughTransfLines(canny, lines1, 0, 180, houthTh);    // prof
        myHoughTransfLines(contours, lines2, 0, 180, houthTh); // opencv

        cv::namedWindow("lines vince", cv::WINDOW_NORMAL);
        cv::imshow("lines vince", lines);

        cv::namedWindow("lines prof", cv::WINDOW_NORMAL);
        cv::imshow("lines prof", lines1);

        cv::namedWindow("lines opencv", cv::WINDOW_NORMAL);
        cv::imshow("lines opencv", lines2);

        //////////////////////////////////*/

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

/* implementazione funzioni */

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

////////////////////////////////////
// Canny

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

    krnl /= sum;

    std::cout << krnl << std::endl;

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
    orient = cv::Mat(src.size(), CV_32FC1);
    for (int v = 0; v < Ix.rows; ++v)
        for (int u = 0; u < Ix.cols; ++u)
            orient.at<float>(v, u) = atan2f(Iy.at<float>(v, u), Ix.at<float>(v, u)) + 2 * CV_PI;

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

void findPeaks3x3(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
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

    return;
}

void findPeaksBilInterpInterp(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
    out = cv::Mat(magn.rows, magn.cols, CV_32FC1, cv::Scalar(0.0));

    float e1 = 0.0, e1x = 0.0, e1y = 0.0, e2 = 0.0, e2x = 0.0, e2y = 0.0;
    float theta = 0.0;

    // pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
    for (int r = 1; r < orient.rows - 1; ++r)
    {
        for (int c = 1; c < orient.cols - 1; ++c)
        {
            theta = orient.at<float>(r, c);

            e1x = c + cos(theta);
            e1y = r + sin(theta);
            e2x = c - cos(theta);
            e2y = r - sin(theta);

            e1 = bilinear<float>(magn, e1y, e1x);
            e2 = bilinear<float>(magn, e2y, e2x);

            if (magn.at<float>(r, c) >= e1 && magn.at<float>(r, c) >= e2)
            {
                out.at<float>(r, c) = magn.at<float>(r, c);
            }
        }
    }

    return;
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

    std::cout << "Optiaml tresholds: tlow: " << tlow << " - thigh: " << thigh
              << std::endl
              << std::endl
              << std::endl;

    return;
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
    out = cv::Mat(magn.rows, magn.cols, magn.type(), cv::Scalar(0.0));

    float tmpVal = 0.0;

    for (int v = 1; v < out.rows - 1; ++v)
    {
        for (int u = 1; u < out.cols - 1; ++u)
        {
            out.at<float>(v, u) = magn.at<float>(v, u);
        }
    }

    out.convertTo(out, CV_8UC1);

    // passata 1

    for (int v = 1; v < out.rows - 1; ++v)
    {
        for (int u = 1; u < out.cols - 1; ++u)
        {
            tmpVal = magn.at<float>(v, u);

            // Over T-high: keep edge
            if (tmpVal >= t2)
                out.at<u_char>(v, u) = 255;

            // Under T-low: remove edge
            else if (tmpVal < t1)
                out.at<u_char>(v, u) = 0;
        }
    }

    // passata 2

    for (int v = 1; v < out.rows - 1; ++v)
    {
        for (int u = 1; u < out.cols - 1; ++u)
        {
            // per ogni pixel di edge avvia la procedura di crescita dei suoi vicini (ricorsiva)
            if (out.at<u_char>(v, u) == 255)
                findAdjRecursive(out, v, u);
        }
    }

    // passata 3: rimozione dei non massimi rimanenti

    for (int v = 1; v < out.rows - 1; ++v)
    {
        for (int u = 1; u < out.cols - 1; ++u)
        {
            if (out.at<u_char>(v, u) != 255)
                out.at<u_char>(v, u) = 0;
        }
    }

    return;
}

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

////////////////////////////////////
// Hough

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

void myHoughTransfLines(const cv::Mat &image, cv::Mat &lines, const int minTheta, const int maxTheta, const int threshold)
{
    if (image.type() != CV_8UC1)
    {
        std::cerr << "ERROR: the image is not CV_8UC1." << std::endl;
        exit(1);
    }

    if (minTheta < 0 || minTheta >= maxTheta)
    {
        std::cerr << "ERROR: the minimum value of theta minTheta is out of the valid range [0, maxTheta)." << std::endl;
        exit(1);
    }

    if (maxTheta <= minTheta || maxTheta > 180)
    {
        std::cerr << "ERROR: the maximum value of theta maxTheta is out of the valid range (minTheta, PI]." << std::endl;
        exit(1);
    }

    int maxVal;

    if (image.rows > image.cols)
        maxVal = image.rows;
    else
        maxVal = image.cols;

    int max_distance = pow(2, 0.5) * maxVal / 2;

    std::vector<int> acc_row(maxTheta - minTheta + 1, 0);
    std::vector<std::vector<int>> accumulator(2 * max_distance, acc_row);

    for (int r = 0; r < image.rows; ++r)
    {
        for (int c = 0; c < image.cols; ++c)
        {
            if (image.at<u_char>(r, c) > 0)
            {
                for (int theta = minTheta; theta <= maxTheta; ++theta)
                {
                    int rho = (c - image.cols / 2) * cos(theta * CV_PI / 180) + (r - image.rows / 2) * sin(theta * CV_PI / 180);

                    ++accumulator[rho + max_distance][theta];
                }
            }
        }
    }

    cv::Mat acc(2 * max_distance, maxTheta - minTheta, CV_8UC1);

    for (int r = 0; r < 2 * max_distance; ++r)
    {
        for (int t = minTheta; t <= maxTheta; ++t)
        {
            acc.at<u_char>(r, t) = accumulator[r][t];
        }
    }

    cv::namedWindow("Accumulator", cv::WINDOW_NORMAL);
    cv::imshow("Accumulator", acc);

    std::cout << "Points: "
              << std::endl
              << std::endl;

    cv::Point start_point, end_point;

    for (int r = 0; r < acc.rows; ++r)
    {
        for (int t = minTheta; t < acc.cols; ++t)
        {
            if (accumulator[r][t] >= threshold)
            {
                myPolarToCartesian(r, t, start_point, end_point, acc.rows, image);
                cv::line(lines, start_point, end_point, cv::Scalar(0, 0, 255), 2, cv::LINE_4);

                std::cout
                    << "Start: (" << start_point.x << ", " << start_point.y << "); "
                    << "End: (" << end_point.x << ", " << end_point.y << ")"
                    << std::endl
                    << std::endl;
            }
        }
    }

    std::cout
        << std::endl
        << std::endl;

    return;
}
