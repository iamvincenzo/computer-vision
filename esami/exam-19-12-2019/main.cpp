// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv4/opencv2/calib3d/calib3d.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>

//////////////////////////////////////////////
/// EX1
//
// Nota la posizione dei 4 angoli della copertina del libro nell'immagine "input.jpg"
// generare la corrispondente immagine vista dall'alto, senza prospettiva.
//
// Si tratta di trovare l'opportuna trasformazione che fa corrispondere la patch di immagine
// input.jpg corrispondente alla copertina del libro con la vista dall'alto della stessa.
//
// Che tipo di trasformazione e'? Come si puo' calcolare con i dati forniti?
//
// E' possibile utilizzare alcune funzioni di OpenCV
//
void WarpBookCover(const cv::Mat &image, cv::Mat &output, const std::vector<cv::Point2f> &corners_src)
{
    std::vector<cv::Point2f> corners_out;

    /*
     * YOUR CODE HERE
     *
     *
     */

    // posizioni note e fissate dei quattro corner della copertina nell'immagine input
    corners_out = {cv::Point2f(0, 0),     // top left
                   cv::Point2f(430, 0),   // top right
                   cv::Point2f(430, 573), // bottom right
                   cv::Point2f(0, 573)};  // bottom left

    Eigen::Matrix<float, 8, 9> A;

    int x1, y1, x2_, y2_;

    for (int i = 0; i < A.rows(); ++i)
    {
        x1 = corners_src[i].x;
        y1 = corners_src[i].y;
        x2_ = corners_out[i].x;
        y2_ = corners_out[i].y;

        if (i % 2 == 0)
            A.row(i) << -x1, -y1, -1, 0, 0, 0, x1 * x2_, y1 * x2_, x2_;
        else
            A.row(i) << 0, 0, 0, -x1, -y1, -1, x1 * y2_, y1 * y2_, y2_;
    }

    cv::Mat ACv;
    cv::eigen2cv(A, ACv);

    cv::Mat D, U, Vt;
    cv::SVD::compute(ACv, Vt);

    std::cout << "ACv: "
              << std::endl
              << ACv << std::endl;

    // Vt.reshape(3, 3); // homograpy

    std::cout << "Vt: "
              << std::endl
              << Vt << std::endl;

    // calcolo l'omografia corrispondente

    // cv::Mat H = cv::findHomography(cv::Mat(corners_out), cv::Mat(corners_src));

    // std::cout << "H:"
    //           << std::endl
    //           << H << std::endl;

    // for (int r = 0; r < output.rows; ++r)
    // {
    //     for (int c = 0; c < output.cols; ++c)
    //     {
    //         cv::Mat p = (cv::Mat_<double>(3, 1) << c, r, 1);

    //         // trasformo da immagine di destinazione a immagine di input
    //         cv::Mat pp = H * p;

    //         // da omogenee a eucliedee
    //         pp /= pp.at<double>(2, 0);

    //         // verifico che la posizione ottenuta sia contenuta nell'immagine di input
    //         if (round(pp.at<double>(0, 0)) >= 0 && round(pp.at<double>(0, 0)) < image.cols && round(pp.at<double>(1, 0)) >= 0 && round(pp.at<double>(1, 0)) < image.rows)
    //         {
    //             // prelevo il pixel dall'immagine di input e lo copio in quella di output
    //             output.at<cv::Vec3b>(r, c) = image.at<cv::Vec3b>(round(pp.at<double>(1, 0)), round(pp.at<double>(0, 0)));
    //         }
    //     }
    // }

    return;
}
/////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////
/// EX2
//
// Applicare il filtro di sharpening visto a lezione
//
// Per le convoluzioni potete usare le funzioni sviluppate per il primo assegnamento
//
//

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
        for (int u = padW; u < padded.cols - padW; ++u)
            padded.at<u_char>(v, u) = src.at<u_char>((v - padH), (u - padW));

    return;
}

void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stride = 1)
{
    if (!checkOddKernel(krn))
        return;

    int padH = (krn.rows - 1) / 2;
    int padW = (krn.cols - 1) / 2;

    cv::Mat padded;
    addZeroPadding(src, padded, padH, padW);

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

void myContrastStretching(const cv::Mat &src, cv::Mat &out)
{
    int minf = 100000, maxf = 0;

    out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

    for (int v = 0; v < out.rows; ++v)
    {
        for (int u = 0; u < out.cols; ++u)
        {
            int val = (int)src.at<u_char>(v, u);

            if (minf > val)
                minf = val;

            if (maxf < val)
                maxf = val;
        }
    }

    for (int v = 0; v < out.rows; ++v)
    {
        for (int u = 0; u < out.cols; ++u)
        {
            int val = (int)src.at<u_char>(v, u);

            out.at<u_char>(v, u) = (int)(val - minf) / (float)(maxf - minf);
        }
    }

    return;
}

void sharpening(const cv::Mat &image, cv::Mat &output, float alpha = 0.8f)
{
    output = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));

    cv::Mat LoG_conv_I;

    /*
     * YOUR CODE HERE
     *
     *
     */

    int size = 3;
    float dataLog[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    cv::Mat logFilter(size, size, CV_32F, dataLog);

    myfilter2D(image, logFilter, LoG_conv_I);

    // display custom convolution result
    cv::Mat outDisplay;
    cv::convertScaleAbs(LoG_conv_I, outDisplay);
    cv::namedWindow("myLoGFilter", cv::WINDOW_NORMAL);
    cv::imshow("myLoGFilter", outDisplay);

    cv::namedWindow("grey image", cv::WINDOW_NORMAL);
    cv::imshow("grey image", image);

    LoG_conv_I *= alpha;

    cv::Mat LoG_conv_I_8U;
    LoG_conv_I.convertTo(LoG_conv_I_8U, CV_8UC1);

    cv::Mat LoG_conv_I_8U_CS;
    myContrastStretching(LoG_conv_I_8U, LoG_conv_I_8U_CS);

    for (int v = 0; v < image.rows; ++v)
    {
        for (int u = 0; u < image.cols; ++u)
        {
            output.at<u_char>(v, u) = image.at<u_char>(v, u) - LoG_conv_I_8U_CS.at<u_char>(v, u);
        }
    }

    return;
}
//////////////////////////////////////////////

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage ./prova <image_filename>" << std::endl;
        return 0;
    }

    // images
    cv::Mat input;

    // load image from file
    input = cv::imread(argv[1]);
    if (input.empty())
    {
        std::cout << "Error loading input image " << argv[1] << std::endl;
        return 1;
    }

    //////////////////////////////////////////////
    /// EX1
    //
    // Creare un'immagine contenente la copertina del libro input come vista "dall'alto" (senza prospettiva)
    //
    //
    //

    // Dimensioni note e fissate dell'immagine di uscita (vista dall'alto):
    constexpr int outwidth = 431;
    constexpr int outheight = 574;
    cv::Mat outwarp(outheight, outwidth, input.type(), cv::Scalar(0));

    // posizioni note e fissate dei quattro corner della copertina nell'immagine input
    std::vector<cv::Point2f> pts_src = {cv::Point2f(274, 189),  // top left
                                        cv::Point2f(631, 56),   // top right
                                        cv::Point2f(1042, 457), // bottom right
                                        cv::Point2f(722, 764)}; // bottom left

    WarpBookCover(input, outwarp, pts_src);
    //////////////////////////////////////////////

    //////////////////////////////////////////////
    /// EX2
    //
    // Applicare uno sharpening all'immagine cover
    //
    // Immagine = Immagine - alfa(LoG * Immagine)
    //
    //
    // alfa e' una costante float, utilizziamo 0.5
    //
    //
    // LoG e' il Laplaciano del Gaussiano. Utilizziamo l'approssimazione 3x3 vista a lezione
    //
    //
    // In questo caso serve fare il contrast stratching nelle convoluzioni?
    //
    //

    // immagine di uscita sharpened
    cv::Mat sharpened(input.rows, input.cols, CV_8UC1);

    // convertiamo l'immagine della copertina a toni di grigio, per semplicita'
    cv::Mat inputgray(input.rows, input.cols, CV_8UC1);
    cv::cvtColor(input, inputgray, cv::COLOR_BGR2GRAY);

    sharpening(inputgray, sharpened, 0.8);
    //////////////////////////////////////////////

    ////////////////////////////////////////////
    /// WINDOWS
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", input);

    cv::Mat outimage_win(std::max(input.rows, outwarp.rows), input.cols + outwarp.cols, input.type(), cv::Scalar(0));
    input.copyTo(outimage_win(cv::Rect(0, 0, input.cols, input.rows)));
    outwarp.copyTo(outimage_win(cv::Rect(input.cols, 0, outwarp.cols, outwarp.rows)));

    cv::namedWindow("Output outwarp", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output outwarp", outwarp);

    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", outimage_win);

    cv::namedWindow("Input Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray", inputgray);

    cv::namedWindow("Input Gray Sharpened", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray Sharpened", sharpened);

    cv::waitKey();

    return 0;
}
