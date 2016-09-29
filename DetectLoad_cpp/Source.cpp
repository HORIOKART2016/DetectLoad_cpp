#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
#include <string>

using namespace std;
using namespace cv;

const string IMG_PATH = "./P7090778.JPG";
const float RESIZE_RATE = 0.3;

/*
 *	�J���[�摜�̃q�X�g�O�����ψꉻ
 *	�[�x3�Ȃ�Ȃ�ł�OK
 */
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

/*
 *	�w��͈͂̉�f�l�𕽋ς���
 */
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

/*
 *	�摜���ōő�̉�f�l��Ԃ�
 */
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
/*
 *  �V�O���C�h�֐��Ɋ�Â��ăR���g���X�g����
 */
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

/*
 *	��_�̉�f�l�Ɋ�Â��ē������Z�o
 *	�F�������Ⴒ���Ⴕ�Ă���_��
 *	
 *	����
 *		Mat src : ���o�Ώۉ摜�iBGR�j
 *		int retLR : �Ԃ�l���i�[����z��.[left, right]�ŕԂ�.
 *		Point center : ��_.���̉�f�l�Ɋ�Â�,�������獶�E�ɕ���T������.
 *		Size size : ��l�𕽋ς���͈͂�,�����T������y�����͈̔�
 */
void DetectLoadWidth(Mat src, int retLR[2], Point criteria, Size size = Size(10,10))
{
	Mat tmp;
	// bgr�̃q�X�g�O�����ψꉻ
	equalizeColorHist(src, tmp);

	Mat pic;
	// hsv��h�̂ݒ��o
	cvtColor(tmp, pic, CV_BGR2HSV);
	vector<Mat> hsv(3);
	split(pic, hsv);
	// h�̃q�X�g�O�����ψꉻ
	equalizeHist(hsv[0], pic);

	// ��ƂȂ��f�l���Z�o
	int criteria_val = calcAverage(pic, criteria, size);
	// ��l�ɑ΂��鍷���v�Z
	int line_height = size.height;
	Mat line_img(line_height, pic.cols - size.width, CV_8U);
	for (int y = criteria.y - line_height / 2; y < criteria.y + (int)(line_height / 2.0 + 0.5); y++)
	{
		for (int x = size.width / 2; x < pic.cols - size.width / 2; x++)
		{
			//int val = calcAverage(pic, Point(x, y), size) - criteria_val;
			int val = pic.data[y * pic.cols + x] - criteria_val;
			line_img.data[(y - criteria.y + line_height / 2) * line_img.cols + x - size.width / 2] = val >= 0 ? val : 256 + val;
		}
	}
	Mat result = contrastSigmoid(line_img); // 2�l���ł�������

	// �m�C�Y����
	morphologyEx(result, line_img, MORPH_CLOSE, Mat(), Point(-1, -1), 3);

	// ���o�Ώۂ̕����Z�o
	retLR[0] = retLR[1] = 0;
	for (int y = 0; y < line_img.rows; y++)
	{
		int x;

		for (x = criteria.x; x > 0; x--)
			if (line_img.data[y * line_img.cols + x] < 200)
				break;
		retLR[0] += x;

		for (x = criteria.x; x < line_img.cols; x++)
			if (line_img.data[y * line_img.cols + x] < 200)
				break;
		retLR[1] += x;
	}
	retLR[1] = retLR[1] / line_img.rows + size.width / 2; // ���ƂŊO��l�΍􂷂�(����)
	retLR[0] = retLR[0] / line_img.rows + size.width / 2;
}

void main()
{
	Mat src = imread(IMG_PATH, 1);
	Mat original_image;
	resize(src, original_image, Size(), RESIZE_RATE, RESIZE_RATE);

	cout << "unko" << endl;
	
	Mat pic;
	// resize
	resize(src, pic, Size(), RESIZE_RATE, RESIZE_RATE);

	Point criteria(pic.cols / 2, pic.rows / 2);
	Size size(10, 10);
	int widthLR[2];
	DetectLoadWidth(pic, widthLR, criteria);

	cout << widthLR[0] << "," << widthLR[1] << endl;;

	// �`��
	line(original_image, Point(0, criteria.y - size.height / 2), Point(original_image.cols, criteria.y - size.height / 2), Scalar(0, 0, 255));
	line(original_image, Point(0, criteria.y + size.height / 2), Point(original_image.cols, criteria.y + size.height / 2), Scalar(0, 0, 255));

	line(original_image, Point(widthLR[1], 0), Point(widthLR[1], original_image.rows), Scalar(0, 0, 255));
	line(original_image, Point(widthLR[0], 0), Point(widthLR[0], original_image.rows), Scalar(0, 0, 255));

	imshow(IMG_PATH, original_image);
	waitKey(0);
}
