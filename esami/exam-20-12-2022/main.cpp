// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

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
    int wait_t;             //!< waiting time
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

        // cv::namedWindow("Gaussian", cv::WINDOW_NORMAL);
        // cv::imshow("Gaussian", blurdisplay);

        cv::Mat magnitude, orientation;
        sobel3x3(blurdisplay, magnitude, orientation);

        // cv::Mat magndisplay;
        // magnitude.convertTo(magndisplay, CV_8UC1);
        // cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
        // cv::imshow("sobel magnitude", magndisplay);

        // cv::Mat ordisplay;
        // orientation.copyTo(ordisplay);
        // float *orp = (float *)ordisplay.data;
        // for (int i = 0; i < ordisplay.cols * ordisplay.rows; ++i)
        //     if (magndisplay.data[i] < 50)
        //         orp[i] = 0;
        // cv::convertScaleAbs(ordisplay, ordisplay, 255 / (2 * CV_PI));
        // cv::Mat falseColorsMap;
        // cv::applyColorMap(ordisplay, falseColorsMap, cv::COLORMAP_JET);
        // cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
        // cv::imshow("sobel orientation", falseColorsMap);

        cv::Mat nms, nmsdisplay;
        findPeaks(magnitude, orientation, nms);
        // nms.convertTo(nmsdisplay, CV_8UC1);
        // cv::namedWindow("edges after NMS", cv::WINDOW_NORMAL);
        // cv::imshow("edges after NMS", nmsdisplay);

        cv::Mat canny;
        if (doubleTh(nms, canny, args.tlow, args.thigh))
        {
            std::cerr << "ERROR: t_low shoudl be lower than t_high" << std::endl;
            exit(1);
        }
        cv::namedWindow("Canny final result", cv::WINDOW_NORMAL);
        cv::imshow("Canny final result", canny);

        //////////////////////////////////
        // HOUGH-PROCESSING VINCE

        cv::Mat blurred1;
        cv::blur(grey, blurred1, cv::Size(3, 3));

        cv::Mat contours;
        cv::Canny(blurred1, contours, 50, 200, 3);

        // display image
        cv::namedWindow("opencv canny", cv::WINDOW_NORMAL);
        cv::imshow("opencv canny", contours);

        cv::Mat lines, lines1;
        image.copyTo(lines);
        image.copyTo(lines1);
        myHoughLines(canny, lines, 0, 180, 150);
        myHoughLines(contours, lines1, 0, 180, 150);

        // display image
        cv::namedWindow("lines prof", cv::WINDOW_NORMAL);
        cv::imshow("lines prof", lines);

        // display image
        cv::namedWindow("lines opencv canny", cv::WINDOW_NORMAL);
        cv::imshow("lines opencv canny", lines1);

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
    // std::cout << "DEBUG: Horizontal Gaussian Kernel:\n"
    //           << hg << "\nSum: " << cv::sum(hg)[0] << std::endl;
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

    // cv::namedWindow("Intermediate Canny result", cv::WINDOW_NORMAL);
    // cv::imshow("Intermediate Canny result", first);
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
