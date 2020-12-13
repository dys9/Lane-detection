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
	//영상을 불러온다
	VideoCapture capture("lane.mp4");

	//마스크 생성
	Mat mask = Mat::zeros(HEIGHT, WIDTH, TYPE_0);
	
	// 마스크를 그릴 점들 생성하여 벡터에 저장
	vector<Point> contour;
	contour.push_back(Point(524, 425));
	contour.push_back(Point(284, 650));
	contour.push_back(Point(1130, 659));
	contour.push_back(Point(659, 425));

	// 벡터의 주소를 저장
	const Point *pts = (const Point*) Mat(contour).data;
	// 백터의 수
	int npts = Mat(contour).rows;
	// 각 점을 흰 색으로 채우고 mask에 저장
	fillPoly(mask, &pts, &npts, 1, (255, 255, 255));

	// forever
	for (;;)
	{
		Mat Img; // 원본 이미지
		if (capture.read(Img) == false) break; // 한 화면을 캡쳐한다.
		Mat Img_gray; // 흑백 영상
		Mat Img_edge; // 테두리 영상
		Mat Img_RoI; // RoI
		Mat Img_road(HEIGHT, WIDTH, TYPE_16); // 도로 영상
		Mat Img_result(HEIGHT, WIDTH, TYPE_16); // 결과 영상

		cvtColor(Img, Img_gray, COLOR_BGR2GRAY); // 흑백영상으로 전환
		bitwise_and(mask, Img_gray, Img_RoI); // 마스크와 흑백영상을 AND 연산 하여, 저장
		Canny(Img_RoI, Img_edge, (Img_gray.rows + Img.cols) / 10, (Img.rows + Img_gray.cols) / 9); // 영상의 테두리를 검출하여 저장

		vector<Vec2f> lines; // 선들을 저장할 벡터
		HoughLines(Img_edge, lines, 1, CV_PI / 180, 40, 0, 0); // 선을 검출하여, 벡터에 저장
		int cnt_P = 0; // 기울기가 양수인 직선
		int cnt_N = 0; // 기울기가 음수인 직선
		Point avg_P_pt1(0, 0), avg_P_pt2(0, 0); // 기울기가 양수인 평균 직선을 그리기 위한 점
		Point avg_N_pt1(0, 0), avg_N_pt2(0, 0); // 기울기가 음수인 평균 직선을 그리기 위한 점
		
		for (size_t i = 0; i < lines.size(); i++) // vector의 요소 만큼 반복
		{
			float rho = lines[i][0]; // 후프변환에서 거리 값 r
			float theta = lines[i][1]; // 후프 변환에서 거리 값 세타
			double a = cos(theta), b = sin(theta); 
			double x0 = a * rho, y0 = b * rho;
			Point pt1, pt2; // 직선의  두 점을 저장.
			pt1.x = cvRound(x0 + (Img_gray.rows + Img_gray.cols) * (-b) );
			pt1.y = 	cvRound(y0 + (Img_gray.rows + Img_gray.cols) * a);
			pt2.x = cvRound(x0 - (Img_gray.rows + Img_gray.cols) * (-b));
			pt2.y = cvRound(y0 - (Img_gray.rows + Img_gray.cols) * a);
			
			//직선의 기울기를 계산
			double encline = ((pt1.y - pt2.y) / double(pt1.x - pt2.x));
			if ( abs(encline) < 0.25 ) //수평의 가까운 직선일 경우
			{// 벡터에서 해당하는 직선을 지우고, 상위 루프로 돌아간다.
				lines.erase(lines.begin() + (int)i);
				continue;
			}
			else if (encline < 0)
			{// 기울기가 음수일 경우의 점들의 좌표의 합
				avg_N_pt1.x += pt1.x;
				avg_N_pt2.x += pt2.x;
				avg_N_pt1.y += pt1.y;
				avg_N_pt2.y += pt2.y;
				cnt_N++;
			}
			else if (encline > 0)
			{// 기울기가 양수일 경우의 점들의 좌표의 합
				avg_P_pt1.x += pt1.x;
				avg_P_pt2.x += pt2.x;
				avg_P_pt1.y += pt1.y;
				avg_P_pt2.y += pt2.y;
				cnt_P++;
			}
		}
		// 기울기가 양수일 때의 점의 평균 위치
		avg_P_pt1.x /= cnt_P;
		avg_P_pt2.x /= cnt_P;
		avg_P_pt1.y /= cnt_P;
		avg_P_pt2.y /= cnt_P;

		// 기울기가 음수일 때의 점의 평균 위치
		avg_N_pt1.x /= cnt_N;
		avg_N_pt2.x /= cnt_N;
		avg_N_pt1.y /= cnt_N;
		avg_N_pt2.y /= cnt_N;

		// 평균 직선을 그린다
		line(Img, avg_P_pt1, avg_P_pt2, (0, 255, 255), 3, LINE_AA);
		line(Img, avg_N_pt1, avg_N_pt2, (0, 255, 255), 3, LINE_AA);

		// 평균 직선의 교점을 구한다.
		Point cross_pt;
		Point dx = avg_P_pt1 - avg_N_pt1;
		Point d1 = avg_N_pt2 - avg_N_pt1;
		Point d2 = avg_P_pt2 - avg_P_pt1;

		float cross = d1.x*d2.y - d1.y*d2.x;
		if (abs(cross) <  1e-8)
			continue;
		
		double t1 = (dx.x * d2.y - dx.y * d2.x)/cross;
		cross_pt = avg_N_pt1 + d1*t1;


		// 교점을 출력한다.
		circle(Img, cross_pt, 3, Scalar(0, 0, 255),10,-1);

		// 위에서 사용한 콘투어를 초기화 한다.
		contour.clear();

		// 도로영역을 표시하기위한 점들을 저장한다.
		contour.push_back(avg_N_pt1);
		contour.push_back(avg_P_pt2);
		contour.push_back(cross_pt);

		// 도로영역을 칠한다.
		const Point *pts = (const cv::Point*) Mat(contour).data;
		int npts = Mat(contour).rows;
		fillPoly(Img_road, &pts, &npts, 1, Scalar(100, 255, 100));

		// 원본 이미지와 도로영역을 합하여 출력한다.
		addWeighted(Img, 0.7, Img_road, 0.3, 0.0, Img_result);
		

		imshow("result", Img_result);
		waitKey(30);
	}
	
}
