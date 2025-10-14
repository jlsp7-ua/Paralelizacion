#include <opencv2/opencv.hpp>
#include <thread>
#include <filesystem>
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

int main(int argc, char* argv[]) {
    if (argc != 2){
        printf("Argumento: [img]");
        return 0;
    }
    Mat img = imread(argv[1]);
    
    if (img.empty()) {
        printf("No se pudo cargar la imagen\n");
        return -1;
    }

    // Crear versiones coloreadas
    Mat red   = colorize(img, Scalar(0, 0, 255));
    Mat green = colorize(img, Scalar(0, 255, 0));
    Mat blue  = colorize(img, Scalar(255, 0, 0));
    Mat yellow= colorize(img, Scalar(0, 255, 255));
    imwrite("resultados/red.png", red);
    imwrite("resultados/green.png", green);
    imwrite("resultados/blue.png", blue);
    imwrite("resultados/yellow.png", yellow);

    // Aplicar bilateral filter a cada una por separado
    Mat red_b, green_b, blue_b, yellow_b;
    bilateralFilter(red, red_b,15, 200, 200);          // bilateral
    GaussianBlur(green, green_b, Size(15,15), 10);
    medianBlur(blue, blue_b, 15); // kernel grande
    Canny(yellow, yellow_b, 100, 200);                      // detección de bordes

// Convertir a 3 canales SOLO si tiene 1 canal
    if (yellow_b.channels()== 1) cvtColor(yellow_b, yellow_b, COLOR_GRAY2BGR);

    imwrite("resultados/redb.png", red_b);
    imwrite("resultados/greenb.png", green_b);
    imwrite("resultados/blueb.png", blue_b);
    imwrite("resultados/yellowb.png", yellow_b);

    // Combinar en cuadrícula 2x2
    Mat top, bottom, final_img;
    hconcat(red_b, green_b, top);
    hconcat(blue_b, yellow_b, bottom);
    vconcat(top, bottom, final_img);
    imwrite("resultados/combined.png", final_img);

    // efecto cómic
    Mat comic;
    comic = comicEffect(final_img);

    imwrite("resultados/resultado_bilateral_comic.png", comic);
    waitKey(0);
    return 0;
}
