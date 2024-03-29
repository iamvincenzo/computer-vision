// std
#include <iostream>
#include <fstream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// utils
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage lab5_2 <image_filename> <camera_params_filename>" << std::endl;
        return 0;
    }

    // output image
    cv::Mat ipm_image(cv::Size(400, 400), CV_8UC3, cv::Scalar(0, 0, 0));

    // load image from file
    cv::Mat input;
    input = cv::imread(argv[1]);

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    imshow("Input", input);
    cv::waitKey(10);

    // load camera params
    CameraParams params;
    LoadCameraParams(argv[2], params);

    Eigen::Matrix<float, 3, 4> K;
    K << params.ku, 0.0, params.u0, 0.0,
        0.0, params.kv, params.v0, 0.0,
        0.0, 0.0, 1.0, 0.0;

    Eigen::Matrix<float, 4, 4> RT;
    cv::Affine3f RT_inv = params.RT.inv();
    RT << RT_inv.matrix(0, 0), RT_inv.matrix(0, 1), RT_inv.matrix(0, 2), RT_inv.matrix(0, 3),
        RT_inv.matrix(1, 0), RT_inv.matrix(1, 1), RT_inv.matrix(1, 2), RT_inv.matrix(1, 3),
        RT_inv.matrix(2, 0), RT_inv.matrix(2, 1), RT_inv.matrix(2, 2), RT_inv.matrix(2, 3),
        0, 0, 0, 1;

    /*
     * YOUR CODE HERE: Choose a planar constraint and realize an inverse perspective mapping
     * 1) compute the perspective matrix P
     * 2) choose a planar constraint (ex. z=0) to be used during the inverse perspective mapping
     * 3) compute the inverse perspective matrix corresponding to the chosen constraint
     * 4) realize the inverse perspective mapping
     */

    // hint: partire da M = K*RT

    Eigen::Matrix<float, 3, 4> M = K * RT;

    Eigen::Matrix<float, 3, 3> M_p;
    M_p << M.coeff(0, 0), M.coeff(0, 2), M.coeff(0, 3),
        M.coeff(1, 0), M.coeff(1, 2), M.coeff(1, 3),
        M.coeff(2, 0), M.coeff(2, 2), M.coeff(2, 3);

    Eigen::Matrix<float, 3, 3> M_p_inv;
    M_p_inv = M_p.inverse();

    std::cout << M_p_inv << std::endl;

    Eigen::Vector3f P;
    Eigen::Vector3f p;

    int scale = 50;
    int t = ipm_image.cols / 2;
    int x, y = 0;

    for (int v = 0; v < input.rows; ++v)
    {
        for (int u = 0; u < input.cols; ++u)
        {
            p << u, v, 1; // Eigen: (x = righe, y = colonne)

            P = M_p_inv * p;

            x = ipm_image.cols / 2 + scale * (P.x() / P.z()); // colonna
            y = ipm_image.rows - scale * (P.y() / P.z()); // riga

            // il sistema di riferimento si piazza al centro in basso

            if (x >= 0 && x < ipm_image.rows && y >= 0 && y < ipm_image.cols)
            {
                ipm_image.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(v, u);
            }
        }
    }

    cv::namedWindow("IPM", cv::WINDOW_AUTOSIZE);
    imshow("IPM", ipm_image);

    std::cout << "Press any key to quit" << std::endl;

    cv::waitKey(0);

    return 0;
}
