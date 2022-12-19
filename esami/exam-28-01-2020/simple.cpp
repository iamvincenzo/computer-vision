// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

// #define USE_OPENCVVIZ
#ifdef USE_OPENCVVIZ
#include <opencv2/viz.hpp>

void PointsToMat(const std::vector<cv::Point3f> &points, cv::Mat &mat)
{
	mat = cv::Mat(1, 3 * points.size(), CV_32FC3);
	for (unsigned int i = 0, j = 0; i < points.size(); ++i, j += 3)
	{
		mat.at<float>(j) = points[i].x;
		mat.at<float>(j + 1) = points[i].y;
		mat.at<float>(j + 2) = points[i].z;
	}
}
#endif

struct points3D
{
	float x;
	float y;
	float z;
} mypoints;

/*
 * Piano passante per tre punti:
 *
 * https://en.wikipedia.org/wiki/Plane_(geometry)
 *
 * Equazione del piano usata: ax + by +cz + d = 0
 */
//
// DO NOT TOUCH
void plane3points(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, float &a, float &b, float &c, float &d)
{
	cv::Point3f p21 = p2 - p1;
	cv::Point3f p31 = p3 - p1;

	a = p21.y * p31.z - p21.z * p31.y;
	b = p21.x * p31.z - p21.z * p31.x;
	c = p21.x * p31.y - p21.y * p31.x;

	d = -(a * p1.x + b * p1.y + c * p1.z);
}

/*
 * Distanza punto piano.
 *
 * https://en.wikipedia.org/wiki/Plane_(geometry)
 */
//
// DO NOT TOUCH
float distance_plane_point(cv::Point3f p, float a, float b, float c, float d)
{
	return fabs((a * p.x + b * p.y + c * p.z + d)) / (sqrt(a * a + b * b + c * c));
}

/////////////////////////////////////////////////////////////////////
//	EX1
//
//	 Calcolare le coordinate 3D x,y,z per ogni pixel, nota la disparita'
//
//	 Si vedano le corrispondenti formule per il calcolo (riga, colonna, disparita') -> (x,y,z)
//
//	 Utilizzare i parametri di calibrazione forniti
//
//	 I valori di disparita' sono contenuti nell'immagine disp
void compute3Dpoints(const cv::Mat &disparity, std::vector<cv::Point3f> &points, std::vector<cv::Point2i> &rc)
{
	// Parametri di calibrazione predefiniti
	constexpr float focal = 657.475;
	constexpr float baseline = 0.3;
	constexpr float u0 = 509.5;
	constexpr float v0 = 247.15;

	for (int r = 0; r < disparity.rows; ++r)
	{
		for (int c = 0; c < disparity.cols; ++c)
		{
			float disparityVal = disparity.at<float>(r, c);

			if (disparityVal > 1)
			{
				float x = (c - u0) * baseline / disparityVal;
				float y = -(r - v0) * baseline / disparityVal;
				float z = -baseline * focal / disparityVal;

				// salvo tutti i punti 3D con z entro i 30m, per semplicita'
				if (std::abs(z) < 30)
				{
					points.push_back(cv::Point3f(x, y, z));
					rc.push_back(cv::Point2i(r, c));
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//	 EX. 2
//
//	 Calcolare con RANSAC il piano che meglio modella i punti forniti
//
//	 Si tratta di implementare un plane fitting con RANSAC:
//
//	 1) scelgo a caso 3 punti da points
//	 2) calcolo il piano con la funzione fornita
//	 3) calcolo la distanza di tutti i punto dal piano, con la funzione fornita
//   4) calcolo gli inliers del modello attuale, salvando punti e coordinate immagine corrispondenti
//
//	 Mi devo salvare il modello che ha piu' inliers
//
//	 Una volta ottenuto il piano, generare un'immagine dei soli inliers
void computePlane(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2i> &rc, std::vector<cv::Point3f> &inliers_best_points, std::vector<cv::Point2i> &inliers_best_rc)
{
	// Parametri di RANSAC:
	int N = 10000;		 // numero di iterazioni
	float epsilon = 0.2; // errore massimo di un inliers

	int numberPoints = points.size();

	// 50% of outlier probability
	float thresholdInliers = (1 - 0.5) * numberPoints;

	srand((unsigned)time(NULL));

	for (int iter = 0; iter < N; ++iter)
	{
		int numberInliers = 0;

		int i = rand() % numberPoints;
		int j = rand() % numberPoints;
		int k = rand() % numberPoints;

		float a = 0;
		float b = 0;
		float c = 0;
		float d = 0;

		cv::Point3f firstPoint = points[i];
		cv::Point3f secondPoint = points[j];
		cv::Point3f thirdPoint = points[k];

		plane3points(firstPoint, secondPoint, thirdPoint, a, b, c, d);

		for (int index = 0; index < numberPoints; ++index)
		{
			cv::Point3f point = points[index];

			if (index == i || index == j || index == k)
				continue;

			float dist = distance_plane_point(point, a, b, c, d);

			if (dist < epsilon)
			{
				inliers_best_points.push_back(point);
				inliers_best_rc.push_back(cv::Point2i(rc[index].x, rc[index].y));
				++numberInliers;
			}
		}

		if (numberInliers >= thresholdInliers)
			break;

		inliers_best_points.clear();
		inliers_best_rc.clear();
	}
}

int main(int argc, char **argv)
{

	//////////////////////////////////////////////////////////////////
	// Parse argument list:
	//
	// DO NOT TOUCH
	if (argc != 3)
	{
		std::cerr << "Usage ./prova <left_image_filename> <dsi_filename" << std::endl;
		return 0;
	}

	// opening left file
	std::cout << "Opening " << argv[1] << std::endl;
	cv::Mat left_image = cv::imread(argv[1], CV_8UC1);
	if (left_image.empty())
	{
		std::cout << "Unable to open " << argv[1] << std::endl;
		return 1;
	}
	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// Lettura delle disparita'
	//
	// DO NOT TOUCH
	cv::Mat imgDisparity16U(left_image.rows, left_image.cols, CV_16U, cv::Scalar(0));
	cv::Mat imgDisparityF32(left_image.rows, left_image.cols, CV_32FC1, cv::Scalar(0));

	// Leggiamo la dsi gia' PRECALCOLATA da file
	std::ifstream dsifile(argv[2], std::ifstream::binary);
	if (!dsifile.is_open())
	{
		std::cout << "Unable to open " << argv[2] << std::endl;
		return 1;
	}
	dsifile.seekg(0, std::ios::beg);
	dsifile.read((char *)imgDisparity16U.data, imgDisparity16U.rows * imgDisparity16U.cols * 2);
	dsifile.close();

	imgDisparity16U.convertTo(imgDisparityF32, CV_32FC1);
	imgDisparityF32 /= 16.0;
	//////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	//	EX1
	//
	//	 Calcolare le coordinate 3D x,y,z per ogni pixel, nota la disparita'
	//
	//	 Si vedano le corrispondenti formule per il calcolo (riga, colonna, disparita') -> (x,y,z)
	//
	//	 Utilizzare i parametri di calibrazione forniti
	//
	//	 I valori di disparita' sono contenuti nell'immagine imgDisparityF32

	// vettore dei punti 3D calcolati a partire disparita'
	std::vector<cv::Point3f> points;
	// vettore delle corrispondenti righe,colonne
	std::vector<cv::Point2i> rc;

	compute3Dpoints(imgDisparityF32, points, rc);
	/////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	//	 EX. 2
	//
	//	 Calcolare con RANSAC il piano che meglio modella i punti forniti
	//
	//	 Si tratta di implementare un plane fitting con RANSAC.
	//

	// vettore degli inliers del modello miglioe
	std::vector<cv::Point3f> inliers_best;

	// vettore delle coordinate (r,c) degli inliers miglioi
	std::vector<cv::Point2i> inliers_best_rc;

	computePlane(points, rc, inliers_best, inliers_best_rc);

	/*
	 * Creare un'immagine formata dai soli pixel inliers
	 *
	 * Nella parte di RANSAC precedente dovro' quindi calcolare, oltre ai punti 3D inliers, anche le loro coordinate riga colonna corrispondenti
	 *
	 * Salvare queste (r,c) nel vettore inliers_best_rc ed utilizzarlo adesso per scrivere l'immagine out con i soli pixel inliers
	 */

	// immagine di uscita che conterra' i soli pixel inliers
	cv::Mat out(left_image.rows, left_image.cols, CV_8UC1, cv::Scalar(0));
	/*
	 * YOUR CODE HERE
	 *
	 * Costruzione immagine di soli inliers
	 *
	 * Si tratta di copia gli inliers dentro out
	 */
	/////////////////////////////////////////////////////////////////////

	for (cv::Point2i coordinate : inliers_best_rc)
		out.at<u_char>(coordinate.x, coordinate.y) = left_image.at<u_char>(coordinate.x, coordinate.y);

		///////////////////////////////////////////////////////////////////////////////////////////////////////
		// display images
		//
		//  DO NOT TOUCH
#ifdef USE_OPENCVVIZ
	cv::viz::Viz3d win("3D view");
	win.setWindowSize(cv::Size(800, 600));

	cv::Mat points_mat;
	PointsToMat(points, points_mat);
	win.showWidget("cloud", cv::viz::WCloud(points_mat));

	std::cout << "Press q to exit" << std::endl;
	win.spin();
#endif

	cv::namedWindow("left image", cv::WINDOW_NORMAL);
	cv::imshow("left image", left_image);

	cv::namedWindow("left image out", cv::WINDOW_NORMAL);
	cv::imshow("left image out", out);

	// wait for key
	cv::waitKey();
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	return 0;
}
