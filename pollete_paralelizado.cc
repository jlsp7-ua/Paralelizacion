#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iostream>

using namespace cv;
namespace fs = std::filesystem;

//Función para colorear una imagen
Mat colorize(const Mat& img, const Scalar& color) {
    Mat colored;
    Mat colorMat(img.size(), img.type(), color);
    addWeighted(img, 0.5, colorMat, 0.5, 0.0, colored);
    return colored;
}

//Efecto cómic
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

//Función worker para procesar cada color en paralelo
void processColor(const Mat& img, const Scalar& color, const std::string& colorName, 
                  Mat& result, Mat& resultFiltered) {
    //Colorear
    result = colorize(img, color);
    imwrite("resultados/" + colorName + ".png", result);
    
    //Aplicar filtros según el color
    if (colorName == "red") {
        bilateralFilter(result, resultFiltered, 15, 200, 200);
    } else if (colorName == "green") {
        GaussianBlur(result, resultFiltered, Size(15, 15), 10);
    } else if (colorName == "blue") {
        medianBlur(result, resultFiltered, 15);
    } else if (colorName == "yellow") {
        Canny(result, resultFiltered, 100, 200);
        //Convertir a 3 canales si tiene 1 canal
        if (resultFiltered.channels() == 1) {
            cvtColor(resultFiltered, resultFiltered, COLOR_GRAY2BGR);
        }
    }
    
    imwrite("resultados/" + colorName + "b.png", resultFiltered);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Argumento: [img]\n");
        return 0;
    }
    
    Mat img = imread(argv[1]);
    if (img.empty()) {
        printf("No se pudo cargar la imagen\n");
        return -1;
    }
    
    std::cout << "Iniciando tarea..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    const std::string OUTPUT_DIR = "resultados";
    try {
        if (!fs::exists(OUTPUT_DIR)) {
            if (fs::create_directory(OUTPUT_DIR)) {
                printf("Directorio '%s' creado con éxito.\n", OUTPUT_DIR.c_str());
            } else {
                printf("No se pudo crear el directorio '%s'.\n", OUTPUT_DIR.c_str());
                return -1;
            }
        }
    } catch (const fs::filesystem_error& e) {
        printf("Error del sistema de archivos al crear el directorio: %s\n", e.what());
        return -1;
    }
    
    //Matrices para almacenar resultados
    Mat red, green, blue, yellow;
    Mat red_b, green_b, blue_b, yellow_b;
    
    //Creamos 4 threads para procesar cada color en paralelo
    std::vector<std::thread> threads;
    
    threads.emplace_back(processColor, std::ref(img), Scalar(0, 0, 255), "red", 
                         std::ref(red), std::ref(red_b));
    threads.emplace_back(processColor, std::ref(img), Scalar(0, 255, 0), "green", 
                         std::ref(green), std::ref(green_b));
    threads.emplace_back(processColor, std::ref(img), Scalar(255, 0, 0), "blue", 
                         std::ref(blue), std::ref(blue_b));
    threads.emplace_back(processColor, std::ref(img), Scalar(0, 255, 255), "yellow", 
                         std::ref(yellow), std::ref(yellow_b));
    
    //Esperamos a que todos los threads terminen
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Procesamiento paralelo completado. Combinando resultados..." << std::endl;
    
    //Combinar en cuadrícula 2x2
    Mat top, bottom, final_img;
    hconcat(red_b, green_b, top);
    hconcat(blue_b, yellow_b, bottom);
    vconcat(top, bottom, final_img);
    imwrite("resultados/combined.png", final_img);
    
    //Efecto cómic
    Mat comic;
    comic = comicEffect(final_img);
    imwrite("resultados/resultado_bilateral_comic.png", comic);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Tiempo total transcurrido: " << duration.count() << " segundos." << std::endl;
    
    return 0;
}