#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
#include <string>
#include <windows.h>

using namespace std;
using namespace cv;

const string IMG_PATH = "P7090822.JPG";
const float RESIZE_RATE = 1.0;

int mode = 1;		//0：カメラ　1：JPG

/*
 *	カラー画像のヒストグラム均一化
 *	深度3ならなんでもOK
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
 *	指定範囲の画素値を平均する
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
 *	画像中で最大の画素値を返す
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
 *  シグモイド関数に基づいてコントラスト調整
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
*	基準点の画素値に基づいて道幅を算出
*	色がごちゃごちゃしてたらダメ
*	(面倒だから縦横で関数分けました.気が向いたらまとめます)
*
*	引数
*		Mat src : 検出対象画像（BGR）
*		int retLR, retTB : 返り値を格納する配列.[left, right]or[top, bottom]で返す.
*		Point center : 基準点.この画素値に基づき,ここから左右に幅を探索する.
*		Size size : 基準値を平均する範囲と,道幅探索する範囲
*/
void getHeight(Mat src, int retTB[2], Point criteria, Size size = Size(10, 10))
{
	// 基準となる画素値を算出
	int criteria_val = calcAverage(src, criteria, size);

	// 基準値に対する差分計算
	int calc_height = src.rows - size.height;
	int calc_width = size.width;
	Mat diff_img(calc_height, calc_width, CV_8U);
	for (int y = size.height / 2; y < src.rows - size.height / 2; y++)
	{
		for (int x = criteria.x - calc_width / 2; x < criteria.x + (int)(calc_width / 2.0 + 0.5); x++)
		{
			//int val = calcAverage(pic, Point(x, y), size) - criteria_val;
			int val = src.data[y * src.cols + x] - criteria_val;
			diff_img.data[(y - size.height / 2) * diff_img.cols + x - criteria.x + calc_width / 2] = abs(val);
		}
	}
	Mat result = contrastSigmoid(diff_img); // 2値化でいいかも

	// ノイズ除去
	morphologyEx(result, diff_img, MORPH_OPEN, Mat(), Point(-1, -1), 3);

	// 検出対象の幅を算出
	retTB[0] = retTB[1] = 0;
	for (int x = 0; x < diff_img.cols; x++)
	{
		int y;

		for (y = criteria.y; y > 0; y--)
			if (diff_img.data[y * diff_img.cols + x] > 200)
				break;
		retTB[0] += y;

		for (y = criteria.y; y < diff_img.rows; y++)
			if (diff_img.data[y * diff_img.cols + x] > 200)
				break;
		retTB[1] += y;
	}
	retTB[1] = retTB[1] / diff_img.cols + size.height / 2; // あとで外れ値対策する(かも)
	retTB[0] = retTB[0] / diff_img.cols + size.height / 2;
}

void getWidth(Mat src, int retLR[2], Point criteria, Size size = Size(10, 10))
{
	// 基準となる画素値を算出
	int criteria_val = calcAverage(src, criteria, size);

	// 基準値に対する差分計算
	int calc_height = size.height;
	int calc_width = src.cols - size.width;
	Mat diff_img(calc_height, calc_width, CV_8U);
	for (int y = criteria.y - calc_height / 2; y < criteria.y + (int)(calc_height / 2.0 + 0.5); y++)
	{
		for (int x = size.width / 2; x < src.cols - size.width / 2; x++)
		{
			//int val = calcAverage(pic, Point(x, y), size) - criteria_val;
			int val = src.data[y * src.cols + x] - criteria_val;
			diff_img.data[(y - criteria.y + calc_height / 2) * diff_img.cols + x - size.width / 2] = abs(val);
		}
	}
	Mat result = contrastSigmoid(diff_img); // 2値化でいいかも

	// ノイズ除去
	morphologyEx(result, diff_img, MORPH_OPEN, Mat(), Point(-1, -1), 3);

	// 検出対象の幅を算出
	retLR[0] = retLR[1] = 0;
	for (int y = 0; y < diff_img.rows; y++)
	{
		int x;

		for (x = criteria.x; x > 0; x--)
			if (diff_img.data[y * diff_img.cols + x] > 200)
				break;
		retLR[0] += x;

		for (x = criteria.x; x < diff_img.cols; x++)
			if (diff_img.data[y * diff_img.cols + x] > 200)
				break;
		retLR[1] += x;
	}
	retLR[1] = retLR[1] / diff_img.rows + size.width / 2; // あとで外れ値対策する(かも)
	retLR[0] = retLR[0] / diff_img.rows + size.width / 2;
}

/*
 *	ヒストグラム均一化した色相画像を返す
 *	
 *	引数
 *		Mat& src : 検出対象画像（BGR）
 *		Mat& dst : 結果格納用
 */
void getHueImage(Mat& src, Mat& dst)
{
	Mat tmp;
	// bgrのヒストグラム均一化
	equalizeColorHist(src, tmp);

	// hsvのhのみ抽出
	cvtColor(tmp, dst, CV_BGR2HSV);
	vector<Mat> hsv(3);
	split(dst, hsv);
	// hのヒストグラム均一化
	equalizeHist(hsv[0], dst);
}

void main()
{


	LARGE_INTEGER freq;
	LARGE_INTEGER start, now;

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	
	Mat src;
	Mat original_image; // 元画像バックアップ

	// 画像読み込み
	if (mode == 0){
		VideoCapture cap(0);

		if (!cap.isOpened()){
				cout << "camera open error\n";
				return;
		}
		Sleep(1500);
		cap >> src;
		original_image = src;
	}

	else if (mode == 1){
		src = imread(IMG_PATH, 1);
		unsigned int height, width;
		height = src.rows;
		width = src.cols;
		cout << "width : " << width << endl;
		cout << "height : " << height << endl;
		resize(src, original_image, Size(), (int)640/width, (int)480/height);
	}

	
	
	
	
	

	cout << "unko" << endl;
	
	Mat pic;
	// resize
	resize(src, pic, Size(), RESIZE_RATE, RESIZE_RATE);

	// 前処理して色相画像に変換
	Mat hue_img;
	getHueImage(pic, hue_img);

	// 縦横幅取得
	Point criteria(pic.cols / 2, pic.rows / 2); // 基準点
	Size size(10, 10);
	int widthLR[2], heightTB[2]; // Left Right, Top Bottom
	getWidth(hue_img, widthLR, criteria, size);
	getHeight(hue_img, heightTB, criteria, size);

	cout << widthLR[0] << "," << widthLR[1] << endl;;

	QueryPerformanceCounter(&now);
	printf("time:%5.5lf", (double)(((now.QuadPart - start.QuadPart) * 1000 / freq.QuadPart)));

	// 横幅描画
	line(original_image, Point(0, criteria.y - size.height / 2), Point(original_image.cols, criteria.y - size.height / 2), Scalar(0, 0, 255));
	line(original_image, Point(0, criteria.y + size.height / 2), Point(original_image.cols, criteria.y + size.height / 2), Scalar(0, 0, 255));

	line(original_image, Point(widthLR[1], 0), Point(widthLR[1], original_image.rows), Scalar(0, 0, 255));
	line(original_image, Point(widthLR[0], 0), Point(widthLR[0], original_image.rows), Scalar(0, 0, 255));

	// 縦幅描画
	line(original_image, Point(0, heightTB[0]), Point(original_image.cols, heightTB[0]), Scalar(255, 0, 255));
	line(original_image, Point(0, heightTB[1]), Point(original_image.cols, heightTB[1]), Scalar(255, 0, 255));

	line(original_image, Point(criteria.x - size.width / 2, 0), Point(criteria.x - size.width / 2, original_image.rows), Scalar(255, 0, 255));
	line(original_image, Point(criteria.x + size.width / 2, 0), Point(criteria.x + size.width / 2, original_image.rows), Scalar(255, 0, 255));

	imshow(IMG_PATH, original_image);
	waitKey(0);
}
