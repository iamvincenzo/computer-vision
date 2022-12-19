//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

#define MIN_DISPARITY 0
#define MAX_DISPARITY 127

struct ArgumentList {
  std::string left_image;		    //left image file name
  std::string right_image;		  //right image file name
  int window_size;              //window size
  int wait_t;                   //waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);
unsigned char openAndWait(const char *windowName, cv::Mat &image, const bool destroyWindow = true);

/**
 * @brief This function is used to compute the disparity using the SAD approach.
 * (Note that it is neither the semi-incremental nor the full-incremental algorithm)
 * 
 * @param left the left image [in]
 * @param right the right image [in]
 * @param w_size the size of the window (odd) [in]
 * @param out the output image [out]
*/
void SADDisparity(const cv::Mat& left, const cv::Mat& right, unsigned short w_size, cv::Mat& out){
  if(left.size() != right.size()){
    std::cerr << "SADDisparity() - ERROR: the left and right images has not the same size." << std::endl;
    exit(1);
  }

  if(w_size % 2 == 0){
    std::cerr << "SADDisparity() - ERROR: the window is not odd in size." << std::endl;
    exit(1);
  }

  /*
    The output is initialized with the same size of the left image (also right is good, because they have the same size) and it is a gray scale image.
  */
  out = cv::Mat::zeros(left.size(), CV_8UC1);

  //with the first 2 for cycles we cycle on the rows and cols of the input images
  for(int r = w_size / 2; r < (left.rows - w_size / 2); ++r){
    for(int c = w_size / 2; c < (left.cols - w_size / 2); ++c){
      /*
        For each point / pixel we compute the SAD and, in the end, we want the minimum value.
        So we initialize minSAD (that contains each time the minimum computed SAD) with the highest possible number.
      */
      unsigned int minSAD = UINT_MAX; 
      int minSAD_d; // minSAD_d contains where (the disparity) we have found the minimum SAD

      //we compute all the possible disparities in the range [MIN_DISPARITY; MAX_DISPARITY] (in our case [0, 127])
      //(c - d) > 1 is needed to avoid exiting the image (we move at max for 127 positions or until we reach the end of the row of the image)
      for(int d = MIN_DISPARITY; d < MAX_DISPARITY && (c - d) > 1; ++d){
        unsigned int SAD = 0; //the computed SAD
        
        //we cycle the w_size x w_size window (dr and dc are the offsets on the rows and cols with respect to the current pixel)
        for(int dr = -w_size / 2; dr <= w_size / 2; ++dr){
          for(int dc = -w_size / 2; dc <= w_size / 2; ++dc){
            int curr_r = r + dr; //the considered row according to the current element of the window
            int curr_left_c = c + dc; //the considered column (in the left image) according to the offset
            int curr_right_c = c - d + dc;  //the considered column (in the right image) according to the offset
            SAD += abs(left.data[(curr_r * left.cols + curr_left_c) * left.elemSize1()] - right.data[(curr_r * right.cols + curr_right_c) * right.elemSize1()]);
          }
        }

        if(SAD < minSAD){
          minSAD = SAD;
          minSAD_d = d;
        }
      }

      out.data[(r * left.cols + c) * out.elemSize1()] = minSAD_d;
    }
  }
}

/**
 * @brief This function is used to compute the disparity using the VDisparity approach.
 * 
 * @param left the left image [in]
 * @param right the right image [in]
 * @param out the output image [out]
*/
void VDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& out){
  out = cv::Mat::zeros(left.rows, MAX_DISPARITY, CV_8UC1);

  for(int r = 0; r < left.rows; ++r){
    for(int c = 0; c < left.cols; ++c){
      for(int d = 0; d < MAX_DISPARITY && (c - d) > 0; ++d){

        if(left.data[(r * left.cols + c)] == right.data[(r * left.cols + c - d)]){
          out.data[(r * MAX_DISPARITY + d)] += 1;
        }

      }
    }
  }
}

/**
 * @brief This function is used to compute the disparity using the VDisparity approach.
 * 
 * @param disparity the disparity image [in]
 * @param out the output image [out]
*/
void VDisparity(const cv::Mat& disparity, cv::Mat& out){
  out = cv::Mat::zeros(disparity.rows, MAX_DISPARITY, CV_8UC1); //the output image has the same number of rows of the disparity image and MAX_DISPARITY + 1 columns

  for(int r = 0; r < disparity.rows; ++r){
    for(int c = 0; c < disparity.cols; ++c){
      out.data[r * MAX_DISPARITY + disparity.data[r * disparity.cols + c]] += 1;
    }
  }
}

int main(int argc, char **argv)
{
  bool exit_loop = false;
  //int imreadflags = cv::IMREAD_COLOR; 
  int imreadflags = cv::IMREAD_GRAYSCALE;

  //////////////////////
  //parse argument list:
  //////////////////////
  ArgumentList args;
  if(!ParseInputs(args, argc, argv)) {
    exit(0);
  }

  std::cout << std::endl << "Lab 07 - Disparity with SAD and V-Disparity approaches" << std::endl;
  
  while(!exit_loop)
  {
    
    //opening files
    std::cout<<"Opening " << args.left_image << std::endl;

    cv::Mat left_image = cv::imread(args.left_image, imreadflags);
    
    if(left_image.empty())
    {
      std::cout<<"Unable to open " << args.left_image << std::endl;
      return 1;
    }

    std::cout<<"Opening " << args.right_image << std::endl;

    cv::Mat right_image = cv::imread(args.right_image, imreadflags);
    if(right_image.empty())
    {
      std::cout<<"Unable to open " << args.right_image << std::endl;
      return 1;
    }

    openAndWait("Left image", left_image, false);
    openAndWait("Right image", right_image, false);

    cv::Mat disparity;

    SADDisparity(left_image, right_image, args.window_size, disparity);

    cv::Mat display_disp = disparity * (255.0 / MAX_DISPARITY);
    
    openAndWait("Disparity (SAD)", display_disp, false);

    cv::Mat v_disparity, v_disparity_direct;

    VDisparity(disparity, v_disparity);
    VDisparity(left_image, right_image, v_disparity_direct);

    openAndWait("V-Disparity", v_disparity, false);
    openAndWait("V-Disparity direct", v_disparity_direct, false);

    //wait for key or timeout
    unsigned char key = cv::waitKey(args.wait_t);
    std::cout << "Key " << int(key) << std::endl;

    //here you can implement some looping logic using key value:
    // - pause
    // - stop
    // - step back
    // - step forward
    // - loop on the same frame

    switch(key)
    {
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
  }

  return 0;
}

unsigned char openAndWait(const char *windowName, cv::Mat &image, const bool destroyWindow){
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::imshow(windowName, image);
	
	unsigned char key = cv::waitKey();
	
    	if(key == 'q')
      		exit(EXIT_SUCCESS);
    	if(destroyWindow)
      		cv::destroyWindow(windowName);
      		
    	return key;	
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
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.wait_t = 0;
  args.window_size = 3;
  
  while ((c = getopt (argc, argv, "hl:r:t:w:")) != -1)
    switch (c)
    {
      case 't':
        args.wait_t = atoi(optarg);
        break;

      case 'l':
        args.left_image = optarg;
        break;

      case 'r':
        args.right_image = optarg;
        break;

      case 'w':
        args.window_size = atoi(optarg);
        break;

      case 'h':
      default:
        std::cout << "Usage: " << argv[0] << " -l <left_image> -r <right_image>" << std::endl;
        std::cout << "To exit:  type q" << std::endl << std::endl;
        std::cout << "Allowed options:" << std::endl <<
          "   -w                       window size (it must be odd) [default = 3]" << std::endl <<
          "   -h                       produces help message" << std::endl << std::endl;
        return false;
    }
  return true;
}

#endif


