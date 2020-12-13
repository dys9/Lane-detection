#include <opencv2\opencv.hpp>
#include <time.h>
#define WIDTH 1280
#define HEIGHT 720
#define TYPE_0 0
#define TYPE_16 16
using namespace std;
using namespace cv;
void main()
{
	//������ �ҷ��´�
	VideoCapture capture("lane.mp4");

	//����ũ ����
	Mat mask = Mat::zeros(HEIGHT, WIDTH, TYPE_0);
	
	// ����ũ�� �׸� ���� �����Ͽ� ���Ϳ� ����
	vector<Point> contour;
	contour.push_back(Point(524, 425));
	contour.push_back(Point(284, 650));
	contour.push_back(Point(1130, 659));
	contour.push_back(Point(659, 425));

	// ������ �ּҸ� ����
	const Point *pts = (const Point*) Mat(contour).data;
	// ������ ��
	int npts = Mat(contour).rows;
	// �� ���� �� ������ ä��� mask�� ����
	fillPoly(mask, &pts, &npts, 1, (255, 255, 255));

	// forever
	for (;;)
	{
		Mat Img; // ���� �̹���
		if (capture.read(Img) == false) break; // �� ȭ���� ĸ���Ѵ�.
		Mat Img_gray; // ��� ����
		Mat Img_edge; // �׵θ� ����
		Mat Img_RoI; // RoI
		Mat Img_road(HEIGHT, WIDTH, TYPE_16); // ���� ����
		Mat Img_result(HEIGHT, WIDTH, TYPE_16); // ��� ����

		cvtColor(Img, Img_gray, COLOR_BGR2GRAY); // ��鿵������ ��ȯ
		bitwise_and(mask, Img_gray, Img_RoI); // ����ũ�� ��鿵���� AND ���� �Ͽ�, ����
		Canny(Img_RoI, Img_edge, (Img_gray.rows + Img.cols) / 10, (Img.rows + Img_gray.cols) / 9); // ������ �׵θ��� �����Ͽ� ����

		vector<Vec2f> lines; // ������ ������ ����
		HoughLines(Img_edge, lines, 1, CV_PI / 180, 40, 0, 0); // ���� �����Ͽ�, ���Ϳ� ����
		int cnt_P = 0; // ���Ⱑ ����� ����
		int cnt_N = 0; // ���Ⱑ ������ ����
		Point avg_P_pt1(0, 0), avg_P_pt2(0, 0); // ���Ⱑ ����� ��� ������ �׸��� ���� ��
		Point avg_N_pt1(0, 0), avg_N_pt2(0, 0); // ���Ⱑ ������ ��� ������ �׸��� ���� ��
		
		for (size_t i = 0; i < lines.size(); i++) // vector�� ��� ��ŭ �ݺ�
		{
			float rho = lines[i][0]; // ������ȯ���� �Ÿ� �� r
			float theta = lines[i][1]; // ���� ��ȯ���� �Ÿ� �� ��Ÿ
			double a = cos(theta), b = sin(theta); 
			double x0 = a * rho, y0 = b * rho;
			Point pt1, pt2; // ������  �� ���� ����.
			pt1.x = cvRound(x0 + (Img_gray.rows + Img_gray.cols) * (-b) );
			pt1.y = 	cvRound(y0 + (Img_gray.rows + Img_gray.cols) * a);
			pt2.x = cvRound(x0 - (Img_gray.rows + Img_gray.cols) * (-b));
			pt2.y = cvRound(y0 - (Img_gray.rows + Img_gray.cols) * a);
			
			//������ ���⸦ ���
			double encline = ((pt1.y - pt2.y) / double(pt1.x - pt2.x));
			if ( abs(encline) < 0.25 ) //������ ����� ������ ���
			{// ���Ϳ��� �ش��ϴ� ������ �����, ���� ������ ���ư���.
				lines.erase(lines.begin() + (int)i);
				continue;
			}
			else if (encline < 0)
			{// ���Ⱑ ������ ����� ������ ��ǥ�� ��
				avg_N_pt1.x += pt1.x;
				avg_N_pt2.x += pt2.x;
				avg_N_pt1.y += pt1.y;
				avg_N_pt2.y += pt2.y;
				cnt_N++;
			}
			else if (encline > 0)
			{// ���Ⱑ ����� ����� ������ ��ǥ�� ��
				avg_P_pt1.x += pt1.x;
				avg_P_pt2.x += pt2.x;
				avg_P_pt1.y += pt1.y;
				avg_P_pt2.y += pt2.y;
				cnt_P++;
			}
		}
		// ���Ⱑ ����� ���� ���� ��� ��ġ
		avg_P_pt1.x /= cnt_P;
		avg_P_pt2.x /= cnt_P;
		avg_P_pt1.y /= cnt_P;
		avg_P_pt2.y /= cnt_P;

		// ���Ⱑ ������ ���� ���� ��� ��ġ
		avg_N_pt1.x /= cnt_N;
		avg_N_pt2.x /= cnt_N;
		avg_N_pt1.y /= cnt_N;
		avg_N_pt2.y /= cnt_N;

		// ��� ������ �׸���
		line(Img, avg_P_pt1, avg_P_pt2, (0, 255, 255), 3, LINE_AA);
		line(Img, avg_N_pt1, avg_N_pt2, (0, 255, 255), 3, LINE_AA);

		// ��� ������ ������ ���Ѵ�.
		Point cross_pt;
		Point dx = avg_P_pt1 - avg_N_pt1;
		Point d1 = avg_N_pt2 - avg_N_pt1;
		Point d2 = avg_P_pt2 - avg_P_pt1;

		float cross = d1.x*d2.y - d1.y*d2.x;
		if (abs(cross) <  1e-8)
			continue;
		
		double t1 = (dx.x * d2.y - dx.y * d2.x)/cross;
		cross_pt = avg_N_pt1 + d1*t1;


		// ������ ����Ѵ�.
		circle(Img, cross_pt, 3, Scalar(0, 0, 255),10,-1);

		// ������ ����� ����� �ʱ�ȭ �Ѵ�.
		contour.clear();

		// ���ο����� ǥ���ϱ����� ������ �����Ѵ�.
		contour.push_back(avg_N_pt1);
		contour.push_back(avg_P_pt2);
		contour.push_back(cross_pt);

		// ���ο����� ĥ�Ѵ�.
		const Point *pts = (const cv::Point*) Mat(contour).data;
		int npts = Mat(contour).rows;
		fillPoly(Img_road, &pts, &npts, 1, Scalar(100, 255, 100));

		// ���� �̹����� ���ο����� ���Ͽ� ����Ѵ�.
		addWeighted(Img, 0.7, Img_road, 0.3, 0.0, Img_result);
		

		imshow("result", Img_result);
		waitKey(30);
	}
	
}
