#include<iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgcodecs\imgcodecs.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <string>
#include <windows.h>
#include <vector>
#include <fstream>
#include <math.h>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
using namespace cv;
Mat Kernel, dst;
Mat Eye_image;
String window_name = "Sobel";
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

//for mouse events
Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;



void GammaCorrection(Mat& src, Mat& dst, float fGamma) {
	CV_Assert(src.data);
	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));
	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
			*it = lut[(*it)];
		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}
		break;
	}
	}
}
Mat1b getOctagon(int M) {
	// M positive and multiple of 3
	CV_Assert((M > 0) && ((M % 3) == 0));
	int k = M / 3;
	int rows = M * 2 + 1;
	int cols = M * 2 + 1;
	Point c(M, M);
	Mat1b strel(rows, cols, uchar(0));
	// Octagon vertices
	//       0-1
	//      /   \
		//     7     2
//     |  c  |
//     6     3
//      \   /
//       5-4
	vector<Point> vertices(8);
	vertices[0].x = c.x - k;
	vertices[0].y = 0;
	vertices[1].x = c.x + k;
	vertices[1].y = 0;
	vertices[2].x = cols - 1;
	vertices[2].y = c.y - k;
	vertices[3].x = cols - 1;
	vertices[3].y = c.y + k;
	vertices[4].x = c.x + k;
	vertices[4].y = rows - 1;
	vertices[5].x = c.x - k;
	vertices[5].y = rows - 1;
	vertices[6].x = 0;
	vertices[6].y = c.y + k;
	vertices[7].x = 0;
	vertices[7].y = c.y - k;
	fillConvexPoly(strel, vertices, Scalar(1));
	return strel;
}
Mat laplacian_func(Mat & img) {
	/// Remove noise by blurring with a Gaussian filter
	Mat gray, abs_dst;
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//cvtColor(img, gray, CV_RGB2GRAY);

	/// Apply Laplace function
	Laplacian(img, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	subtract(img, abs_dst, abs_dst);

	imshow("result", abs_dst);

	waitKey(0);
	return abs_dst;
}
Mat Morphology_Operations(Mat &src)
{
	int selection;
	cout << "Select the type of operation:\n" << "0. Opening  \n1.Closing \n2.Morphological Gradient \n3.Top Hat \n4.Black Hat  \n-> ";
	cin >> morph_operator;
	// Since MORPH_X : 2,3,4,5 and 6
	int operation = morph_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(4 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(src, dst, operation, element);
	imshow(window_name, dst);
	waitKey(0);
	return dst;
}
Mat Prewitt(Mat & img) {

	int Hprewitt[3][3] = { { -1, 0, 1 },{ -1, 0, 1 },{ -1, 0, 1 } };
	int Vprewitt[3][3] = { { -1, -1, -1 },{ 0, 0, 0 },{ 1, 1, 1 } };
	int tempInput[3][3];
	int tempPixel = 0;
	Mat src_gray;
	Mat grad;
	const char* window_name2 = "Prewitt";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	int computedIntensity;
	namedWindow(window_name2, WINDOW_AUTOSIZE);
	Mat HprewittMat(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(img, src_gray, COLOR_RGB2GRAY);
	namedWindow(window_name, WINDOW_AUTOSIZE);
	Scalar intensity = img.at<uchar>(Point(50, 50)); // this is how to access intensity at a certain pixel
	Vec3b scalarTempPixel = img.at<Vec3b>(Point(1, 1));

	cout << "Pixel (50,50) has intensity: " << intensity.val[0] << endl;


	// applying horizontal prewitt operator
	cout << "\n Image has resolution: " << img.cols << "x" << img.rows << "\n";

	for (int i = 2; i < img.cols - 1; i++) { // currently going from column 2 to n-2, same for row
		for (int j = 2; j < img.rows - 1; j++) {
			// storing a temporary 3x3 input matrix centered on the current pixel
			//  cout << "Matrix centered on pixel: [" << i << "," << j << "] \n";
			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					intensity = img.at<uchar>(Point(i + k, j + l));
					tempInput[k + 1][l + 1] = intensity.val[0];
					//  cout << "[" << intensity.val[0] << "]";
				}
				//      cout << " \n";
			}
			// convolution of horizontal prewitt kernel with current 3x3 matrix
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					tempPixel = tempPixel + tempInput[x][y] * Hprewitt[x][y];
				}
			}

			scalarTempPixel[0] = tempPixel;
			HprewittMat.at<Vec3b>(Point(i, j)) = scalarTempPixel;
		}
	}
	return HprewittMat;
}
Mat Sobel(Mat& src) {
	Mat src_gray;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	GaussianBlur(src, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	//cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create window
	//namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return grad;
}
Mat Kirch(cv::Mat &img) {
	Mat Kern_x = (Mat_<double>(5, 5) <<
		9, 9, 9, 9, 9,
		9, 5, 5, 5, 9,
		-7, -3, 0, -3, -7,
		-7, -3, -3, -3, -7,
		-7, -7, -7, -7, -7) / 25;
	cout << "hi";
	Mat Kern_45 = (Mat_<double>(5, 5) <<
		9, 9, 9, 9, -7,
		9, 5, 5, -3, -7,
		9, 5, 0, -3, -7,
		9, -3, -3, -3, -7,
		-7, -7, -7, -7, -7) / 25;
	cout << "hi";
	Mat Kern_y = (Mat_<double>(5, 5) <<
		9, 9, -7, -7, -7,
		9, 5, -3, -3, -7,
		9, 5, 0, -3, -7,
		9, 5, -3, -3, -7,
		9, 9, -7, -7, -7) / 25;
	cout << "hi";
	Mat Kern_135 = (Mat_<double>(5, 5) <<
		-7, -7, -7, -7, -7,
		9, -3, -3, -3, -7,
		9, 5, 0, -3, -7,
		9, 5, 5, -3, -7,
		9, 9, 9, 9, -7) / 25;
	cout << "hi";
	Mat Kern_xx = (Mat_<double>(5, 5) <<
		-7, -7, -7, -7, -7,
		-7, -3, -3, -3, -7,
		-7, -3, 0, -3, -7,
		9, 5, 5, 5, 9,
		9, 9, 9, 9, 9) / 25;
	cout << "hi";
	Mat Kern_225 = (Mat_<double>(5, 5) <<
		-7, -7, -7, -7, -7,
		-7, -3, -3, -3, 9,
		-7, -3, 0, 5, 9,
		-7, -3, 5, 5, 9,
		-7, 9, 9, 9, 9) / 25;
	cout << "hi";
	Mat Kern_yy = (Mat_<double>(5, 5) <<
		-7, -7, -7, 9, 9,
		-7, -3, -3, 5, 9,
		-7, -3, 0, 5, 9,
		-7, -3, -3, 5, 9,
		-7, -7, -7, 9, 9) / 25;
	cout << "hi";
	Mat Kern_315 = (Mat_<double>(5, 5) <<
		-7, 9, 9, 9, 9,
		-7, -3, 5, 5, 9,
		-7, -3, 0, 5, 9,
		-7, -3, -3, -3, 9,
		-7, -7, -7, -7, -7) / (25);
	Mat filter_x, filter_45, filter_y, filter_135, filter_xx, filter_225, filter_yy, filter_315, filter;
	//normalize(img, img, 0.0, 1.0, NORM_MINMAX);
	//cvtColor(img, img, CV_BGR2GRAY);
	filter2D(img, filter_x, -1, Kern_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_45, -1, Kern_45, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_y, -1, Kern_y, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_135, -1, Kern_135, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_xx, -1, Kern_xx, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_225, -1, Kern_225, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_yy, -1, Kern_yy, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(img, filter_315, -1, Kern_315, Point(-1, -1), 0, BORDER_DEFAULT);
	bitwise_or(filter_x, filter_45, filter);
	bitwise_or(filter, filter_y, filter);
	bitwise_or(filter, filter_135, filter);
	bitwise_or(filter, filter_xx, filter);
	bitwise_or(filter, filter_225, filter);
	bitwise_or(filter, filter_yy, filter);
	bitwise_or(filter, filter_315, filter);
	cout << "hi";
	return filter;
}
void pixel_value(Mat & src) {
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			cout << (int)src.at<uchar>(i, j) << "\t";
		}
		cout << endl;
	}
}

Mat histogram_matching(Mat & src, Mat & dest)
{
	//Mat result = Mat::zeros(Size(src.cols, src.rows), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			dest.at<float>(i, j) = (src.at<float>(i, j) + dest.at<float>(i - 1, j)) / 2.0;
		}
	}
	return dest;
}

Mat Convolution(cv::Mat &img, cv::Mat &Kern) {
	Mat image;
	// Adding the countour of nulls around the original image, to avoid border problems during convolution
	Mat img_conv = Mat::Mat(image.rows + Kern.rows - 1, image.cols + Kern.cols - 1, CV_64FC3, CV_RGB(0, 0, 0));
	for (int x = 0; x < image.rows; x++) {
		for (int y = 0; y < image.cols; y++) {
			img_conv.at<Vec3d>(x + 1, y + 1)[0] = image.at<Vec3d>(x, y)[0];
			img_conv.at<Vec3d>(x + 1, y + 1)[1] = image.at<Vec3d>(x, y)[1];
			img_conv.at<Vec3d>(x + 1, y + 1)[2] = image.at<Vec3d>(x, y)[2];
		}
	}
	Mat img_color;
	//Performing the convolution
	Mat my_conv = Mat::Mat(img.rows, img.cols, CV_8U, CV_RGB(0, 0, 0));
	for (int x = (Kern.rows - 1) / 2; x < img_conv.rows - ((Kern.rows - 1) / 2); x++) {
		for (int y = (Kern.cols - 1) / 2; y < img_conv.cols - ((Kern.cols - 1) / 2); y++) {
			double comp_1 = 0;
			double comp_2 = 0;
			double comp_3 = 0;
			for (int u = -(Kern.rows - 1) / 2; u <= (Kern.rows - 1) / 2; u++) {
				for (int v = -(Kern.cols - 1) / 2; v <= (Kern.cols - 1) / 2; v++) {
					comp_1 = comp_1 + (img_conv.at<Vec3d>(x + u, y + v)[0] * Kern.at<double>(u + ((Kern.rows - 1) / 2), v + ((Kern.cols - 1) / 2)));
					comp_2 = comp_2 + (img_conv.at<Vec3d>(x + u, y + v)[1] * Kern.at<double>(u + ((Kern.rows - 1) / 2), v + ((Kern.cols - 1) / 2)));
					comp_3 = comp_3 + (img_conv.at<Vec3d>(x + u, y + v)[2] * Kern.at<double>(u + ((Kern.rows - 1) / 2), v + ((Kern.cols - 1) / 2)));
				}
			}
			my_conv.at<Vec3d>(x - ((Kern.rows - 1) / 2), y - (Kern.cols - 1) / 2)[0] = comp_1;
			my_conv.at<Vec3d>(x - ((Kern.rows - 1) / 2), y - (Kern.cols - 1) / 2)[1] = comp_2;
			my_conv.at<Vec3d>(x - ((Kern.rows - 1) / 2), y - (Kern.cols - 1) / 2)[2] = comp_3;
		}
	}
	return my_conv;
}
void equalizeHistWithMask(const Mat1b& src, Mat1b& dst, Mat1b mask = Mat1b())
{
	int cnz = countNonZero(mask);
	if (mask.empty() || (cnz == src.rows*src.cols))
	{
		equalizeHist(src, dst);
		return;
	}

	dst = src.clone();

	// Histogram
	vector<int> hist(256, 0);
	for (int r = 0; r < src.rows; ++r) {
		for (int c = 0; c < src.cols; ++c) {
			if (mask(r, c)) {
				hist[src(r, c)]++;
			}
		}
	}

	// Cumulative histogram
	float scale = 255.f / float(cnz);
	vector<uchar> lut(256);
	int sum = 0;
	for (int i = 0; i < hist.size(); ++i) {
		sum += hist[i];
		lut[i] = saturate_cast<uchar>(sum * scale);
	}

	// Apply equalization
	for (int r = 0; r < src.rows; ++r) {
		for (int c = 0; c < src.cols; ++c) {
			if (mask(r, c)) {
				dst(r, c) = lut[src(r, c)];
			}
		}
	}
}

Mat Channel(Mat & img) {
	int selection;
	cout << "Select the type of kernel:\n" << "1. RED \n2. GREEN \n3. BLUE \n-> ";
	cin >> selection;
	Mat channels[3];
	split(img, channels);
	Mat g = Mat(img.rows, img.cols, CV_8UC1);
	//channels[0] = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);
	//channels[2] = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);


	Mat component;
	switch (selection) {
	case 1: {
		component = channels[0];
		imshow("result_1", component);
		waitKey(0);

		cout << "hi";
		break;
	}

	case 2: {
		component = channels[1];
		imshow("result_1", component);
		waitKey(0);

		cout << "hi";
		break;
	}

	case 3: {
		component = channels[2];
		imshow("result_1", component);
		waitKey(0);

		cout << "hi";
		break;
	}

	default: {
		cerr << "Invalid selection";
	}

	}
	return component;
}
Mat setKernel() {
	Mat Kern;
	int kernel_size;   // permitted sizes: 3, 5, 7, 9 etc
					   //cout << "Select the size of kernel (it should be an odd number from 3 onwards): \n" << endl;
					   //cin >> kernel_size;
	kernel_size = 5;
	// Defining the kernel here
	int selection;
	cout << "Select the type of kernel:\n" << "1. Median \n2. disk \n3. diamond Structure \n4. Lee Sigma Filter \n5. Octagan Structure\n-> ";
	cin >> selection;
	switch (selection) {
	case 1:
		Kern = (Mat_<char>(5, 5) <<
			-1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1,
			-1, -1, 30, -1, -1,
			-1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1);
		break;
	case 2:
	{
		Kern = (Mat_<double>(kernel_size, kernel_size) <<
			0, 0, 1, 0, 0,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			1, 1, 1, 1, 1,
			0, 0, 1, 0, 0) / (kernel_size * kernel_size);
		break;
	}

	case 3: {
		Kern = (Mat_<char>(kernel_size, kernel_size) <<
			0, 0, 1, 0, 0,
			0, 1, 1, 1, 0,
			1, 1, 1, 1, 1,
			0, 1, 1, 1, 0,
			0, 0, 1, 0, 0);
		break;
	}

	case 4: {
		Kern = (Mat_<double>(kernel_size, kernel_size) << 1, 0, 0, 0, 1,
			1, 0, 1, 0, 1,
			1, 0, 11, 0, 1,
			1, 0, -1, 0, 1,
			1, 0, 1, 0, 1);
		break;
	}

	case 5: {
		Kern = getOctagon(9);
		break;
	}

	default:
		cerr << "Invalid selection";
		break;
	}
	cout << "my kernel:\n " << Kern << endl;
	return Kern;
}
/** @function thresh_callback */
void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, 150, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}
Mat thresh_white(Mat filter)
{
	Mat imgThreshold;
	cv::inRange(filter, cv::Scalar(0, 0, 200, 0), cv::Scalar(180, 255, 255, 0), imgThreshold);

	return imgThreshold;
}
int main() {
	vector<String> fb;
	string img_arr = "G:/KMC Banglore/Moderate/*.jpg";
	Mat BilateralFilter, filter, sobel, morp_res1, morp_res, morph, Laplace, prewitt, channel, KERNEL, Gamma;
	glob(img_arr, fb);
	std::string s;
	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);
	for (int i = 0; i < fb.size(); ++i) {
		Eye_image = imread(fb[i], CV_32F);
		s = std::to_string(i);
		if (Eye_image.empty())
		{
			cout << "error in reading the image\n";
			return -1;
		}
		Mat frame;
		Mat ycbcr;
		cvtColor(Eye_image, ycbcr, CV_BGR2YCrCb);


		/*
		Mat chan[3];
		split(ycbcr, chan);

		Mat y = chan[0];
		Mat cb = chan[1];
		Mat cr = chan[2];
		Mat channels[3];
		split(Eye_image, channels);
		imshow("y", y);
		imshow("cb", cb);
		imshow("cr", cr);
		imshow("R", channels[0]);
		imshow("G", channels[1]);
		imshow("B", channels[2]);
		cout << Eye_image.rows;
		for (int i = 0; i < Eye_image.rows;i++) {
		for (int j = 0; j < Eye_image.cols;j++) {
		if (i > Eye_image.rows / 4) {
		Eye_image.at<uchar>(i,j) = 0;
		}

		}
		}
		*/
		/*---------*/
		/*Mat new_image = Mat::zeros(Eye_image.size(), Eye_image.type());

		double alpha; /**< Simple contrast control */
		//int beta;  /**< Simple brightness control */
		/// Initialize values
		/*std::cout << " Basic Linear Transforms " << std::endl;
		std::cout << "-------------------------" << std::endl;
		std::cout << "* Enter the alpha value [1.0-3.0]: "; std::cin >> alpha;
		std::cout << "* Enter the beta value [0-100]: "; std::cin >> beta;

		/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
		for (int y = 0; y < Eye_image.rows; y++)
		{
		for (int x = 0; x < Eye_image.cols; x++)
		{
		for (int c = 0; c < 3; c++)
		{
		new_image.at<Vec3b>(y, x)[c] =
		saturate_cast<uchar>(alpha*(Eye_image.at<Vec3b>(y, x)[c]) + beta);
		}
		}
		}
		*/
		/*--------------------*/
		//Eye_image.convertTo(Eye_image, -1, 2, 0);  //increase (double) contrast
		vector<Mat> bgr_planes;
		split(Eye_image, bgr_planes);
		src = Eye_image;
		/// Establish the number of bins
		int histSize = 256;

		/// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 };
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
		imshow("redCompo", bgr_planes[0]);
		src_gray = bgr_planes[0];
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
		//cout << g_hist << endl;
		//imshow("blueCompo", bgr_planes[2]);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		/// Draw for each channel
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		/// Display
		namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		imshow("calcHist Demo", histImage);

		/// Create Windows
		namedWindow("Original Image", 1);
		namedWindow("New Image", 1);
		int a[20][20];
		/// Show stuff
		imshow("Original Image", Eye_image);
		cout << Eye_image.rows << endl;
		cout << Eye_image.cols << endl;
		src_gray.convertTo(src_gray, CV_8UC3);
		/// Wait until user press some key
		Mat image2 = src_gray.clone();

		// define bounding rectangle
		cv::Rect rectangle(0, 0, src_gray.cols, src_gray.rows - 2);

		cv::Mat result; // segmentation result (4 possible values)
		cv::Mat bgModel, fgModel; // the models (internally used)

								  // GrabCut segmentation
		cv::grabCut(Eye_image,    // input image
			result,   // segmentation result
			rectangle,// rectangle containing foreground
			bgModel, fgModel, // models
			1,        // number of iterations
			cv::GC_INIT_WITH_RECT); // use rectangle
		cout << "oks pa dito" << endl;
		// Get the pixels marked as likely foreground
		cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		// Generate output image
		cv::Mat foreground(src_gray.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::Mat background(image.size(),CV_8UC3,cv::Scalar(255,255,255));
		src_gray.copyTo(foreground, result); // bg pixels not copied

		Mat background;
		GammaCorrection(foreground, src_gray, .5);
		//cv::erode(foreground, foreground, cv::Mat());
		//cv::dilate(foreground, foreground, cv::Mat());		
		cv::absdiff(src_gray, bgr_planes[0], src_gray);
		imshow("foreground", src_gray);
		int kernel_length = 9;
		Mat dst_gray, diff;
		Mat kernel = getOctagon(9);
		src_gray.convertTo(src_gray, CV_8UC1);
		bilateralFilter(src_gray, dst_gray, kernel_length, kernel_length * 2, kernel_length / 2);
		cv::absdiff(bgr_planes[0], dst_gray, diff);
		cv::morphologyEx(dst_gray, morp_res, cv::MORPH_CLOSE, kernel);
		imshow("Morphological CLose " + s, morp_res);
		//get foreground
		morp_res = morp_res - bgr_planes[0];
		//display foreground
		imshow("morphed foreground " + s, morp_res);

		Mat imgThreshold;

		Mat Kern = (Mat_<char>(5, 5) <<
			0, 0, 1, 0, 0,
			0, 0, 1, 0, 0,
			1, 1, 1, 1, 1,
			0, 0, 1, 0, 0,
			0, 0, 1, 0, 0);


		//median filtering -> median filtered (brightened) (fore),i.  image
		Mat filter;
		filter2D(morp_res, filter, -1, Kern, Point(-1, -1), 0, BORDER_DEFAULT);
		imshow("Median Filtering", filter);

		imgThreshold = thresh_white(filter);
		imshow("imgThreshold_1", imgThreshold);
		//filter - green compo  ---to reduce noise
		filter = filter - bgr_planes[0];
		imshow("Less Noised", filter);

		GammaCorrection(filter, filter, 2.0);
		imshow("Enhanced by gamma", filter);


		filter2D(filter, filter, -1, Kern, Point(-1, -1), 0, BORDER_DEFAULT);
		imshow("(gamma)Enhanced with filter", filter);

		//filter - background
		filter = filter - bgr_planes[0];
		imshow("Foregd Vessels", filter);

		imgThreshold = thresh_white(filter);
		cv::morphologyEx(imgThreshold, imgThreshold, cv::MORPH_CLOSE, kernel);
		//Eye_image.copyTo(imgThreshold, imgThreshold);
		//imgThreshold = Eye_image - imgThreshold;
		imshow("imgThreshold", imgThreshold);

		//Eye_image.copyTo(imgThreshold);
		//diff= Kirch(dst_gray)
		//bitwise_not(diff, diff);
		//Mat img_bw = diff < 128;
		Mat img_hist_equalized;
		//equalizeHist(diff, img_hist_equalized); //equalize the histogram
		//imshow("EquiHist", img_hist_equalized);
		//cvtColor(diff, diff, CV_GRAY2RGB);
		//cvtColor(diff, diff, CV_RGB2HSV);


		imshow("Thresholded Image", diff); //show the thresholded image
										   //imshow("blur", diff);
										   // draw rectangle on original image
										   //imshow("foreword", fore);
		Eye_image.copyTo(background, ~result);								  // draw rectangle on original image
		imshow("background", background);
		cv::rectangle(src_gray, rectangle, cv::Scalar(255, 255, 255), 1);

		imshow("eye", src_gray);
		waitKey(0);
		/// Default start
		//morph=Morphology_Operations(Laplace);
		//dst.convertTo(dst, CV_8UC3);
		// ok, now try different kernel
		//	imshow("sigma1", a);
		//cv::absdiff(Eye_image, filter, filter);
		Sleep(100);

	}
}