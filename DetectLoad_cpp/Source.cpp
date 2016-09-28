#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
#include <string>

using namespace std;
using namespace cv;

const string IMG_PATH = "./P7090787.JPG";
const float RESIZE_RATE = 0.3;

void equalizeColorHist(Mat& src, Mat& dst)
{
	vector<Mat> planes(3);
	vector<Mat> output_planes(3);
	vector<Mat> rgb;

	split(src, planes); // bgr

	equalizeHist(planes[0], output_planes[0]);
	equalizeHist(planes[1], output_planes[1]);
	equalizeHist(planes[2], output_planes[2]);

	rgb.push_back(output_planes[0]);
	rgb.push_back(output_planes[1]);
	rgb.push_back(output_planes[2]);

	merge(rgb, dst);
}

int calcAverage(Mat& src, Point center, Size size)
{
	int val = 0;
	for (int y = center.y - size.height / 2; y <= center.y + size.height / 2; y++)
	{
		for (int x = center.x - size.width / 2; x <= center.x + size.width / 2; x++)
		{
			val += src.data[y * src.cols + x];
		}
	}

	return val / (size.width*size.height);
}

int maxBrightness(Mat pic)
{
	int maxBrightness = 0;

	for (int y = 0; y < pic.rows - 1; y++)
	{
		for (int x = 0; x < pic.cols - 1; x++)
		{
			if (pic.data[y * pic.cols + x] > maxBrightness) maxBrightness = pic.data[y * pic.cols + x];
		}
	}
	return maxBrightness;
}

Mat contrastSigmoid(Mat &pic)
{
	float gain = 50;
	int maxbrightness = maxBrightness(pic);

	uchar lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = 255.0 / (1 + exp(-gain * (i - maxbrightness / 2) / maxbrightness));
	}

	Mat p = pic.reshape(0, 1).clone();
	for (int i = 0; i < p.cols; i++)
	{
		p.at<uchar>(0, i) = lut[p.at<uchar>(0, i)];
	}
	return p.reshape(0, pic.rows).clone();

}
void main()
{
	Mat src = imread(IMG_PATH, 1);
	Mat original_image;
	resize(src, original_image, Size(), RESIZE_RATE, RESIZE_RATE);
	Mat tmp;
	//equalizeHist(src, tmp);
	//src = tmp.clone();

	equalizeColorHist(src, tmp);
	src = tmp.clone();

	cvtColor(src, tmp, CV_BGR2HSV);
	vector<Mat> hsv(3);
	split(tmp, hsv);
	src = hsv[0].clone();
	equalizeHist(src, tmp);
	src = tmp.clone();

	Mat pic;
	resize(src, pic, Size(), RESIZE_RATE, RESIZE_RATE);

	Size size(10, 10);
	Point center(pic.cols / 2, pic.rows / 2 + 110);
	int center_val = calcAverage(pic, center, size);
	int line_height = 10;
	Mat line_img(line_height, pic.cols - size.width, CV_8U);
	for (int y = center.y - line_height / 2; y < center.y + (int)(line_height / 2.0 + 0.5); y++)
	{
		for (int x = size.width / 2; x < pic.cols - size.width / 2; x++)
		{
			//int val = calcAverage(pic, Point(x, y), size) - center_val;
			int val = pic.data[y * pic.cols + x] - center_val;
			line_img.data[(y - center.y + line_height / 2) * line_img.cols + x - size.width / 2] = val >= 0 ? val : 256 + val;
		}
	}
	Mat result = contrastSigmoid(line_img); // 2’l‰»‚Å‚¢‚¢‚©‚à

	morphologyEx(result, line_img, MORPH_CLOSE, Mat());
	int right_x = 0;
	int left_x = 0;
	for (int y = 0; y < line_img.rows; y++)
	{
		for (int x = center.x; x < line_img.cols; x++)
		{
			if (line_img.data[y * line_img.cols + x] < 200)
			{
				right_x += x;
				break;
			}
		}
		for (int x = center.x; x > 0; x--)
		{
			if (line_img.data[y * line_img.cols + x] < 200)
			{
				left_x += x;
				break;
			}
		}
	}
	right_x = right_x / line_img.rows + size.width / 2; // ‚ ‚Æ‚ÅŠO‚ê’l‘Îô‚·‚é
	left_x = left_x / line_img.rows + size.width / 2;

	cout << left_x << "," << right_x << endl;;

	line(original_image, Point(0, center.y - line_height / 2), Point(original_image.cols, center.y - line_height / 2), Scalar(0, 0, 255));
	line(original_image, Point(0, center.y + line_height / 2), Point(original_image.cols, center.y + line_height / 2), Scalar(0, 0, 255));

	line(original_image, Point(right_x, 0), Point(right_x, original_image.rows), Scalar(0, 0, 255));
	line(original_image, Point(left_x, 0), Point(left_x, original_image.rows), Scalar(0, 0, 255));

	imshow(IMG_PATH, original_image);
	waitKey(0);
}
