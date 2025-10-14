#include <opencv2/opencv.hpp>
#include <thread>
using namespace cv;

// Función para colorear una imagen
Mat colorize(const Mat& img, const Scalar& color) {
    Mat colored;
    Mat colorMat(img.size(), img.type(), color);
    addWeighted(img, 0.5, colorMat, 0.5, 0.0, colored);
    return colored;
}

// Efecto cómic
Mat comicEffect(const Mat& img) {
    Mat gray, edges, color, comic;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 7);
    Laplacian(gray, edges, CV_8U, 5);
    threshold(edges, edges, 80, 255, THRESH_BINARY_INV);

    bilateralFilter(img, color, 9, 150, 150);

    comic = color.clone();
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) == 0)
                comic.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
        }
    }
    return comic;
}

int main() {
    Mat img = imread("alexelcapo.jpg");
    if (img.empty()) {
        printf("No se pudo cargar la imagen\n");
        return -1;
    }

    // Crear versiones coloreadas
    Mat red   = colorize(img, Scalar(0, 0, 255));
    Mat green = colorize(img, Scalar(0, 255, 0));
    Mat blue  = colorize(img, Scalar(255, 0, 0));
    Mat yellow= colorize(img, Scalar(0, 255, 255));
    imwrite("red.png", red);
    imwrite("green.png", green);
    imwrite("blue.png", blue);
    imwrite("yellow.png", yellow);

    // Aplicar bilateral filter a cada una por separado
    Mat red_b, green_b, blue_b, yellow_b;
    bilateralFilter(red,    red_b,    9, 150, 150);
    bilateralFilter(green,  green_b,  9, 150, 150);
    bilateralFilter(blue,   blue_b,   9, 150, 150);
    bilateralFilter(yellow, yellow_b, 9, 150, 150);
    imwrite("redb.png", red_b);
    imwrite("greenb.png", green_b);
    imwrite("blueb.png", blue_b);
    imwrite("yellowb.png", yellow_b);

    // Combinar en cuadrícula 2x2
    Mat top, bottom, final_img;
    hconcat(red_b, green_b, top);
    hconcat(blue_b, yellow_b, bottom);
    vconcat(top, bottom, final_img);
    imwrite("combined.png", final_img);

    // Blur global + efecto cómic
    Mat comic;
    comic = comicEffect(final_img);

    imwrite("resultado_bilateral_comic.png", comic);
    waitKey(0);
    return 0;
}
