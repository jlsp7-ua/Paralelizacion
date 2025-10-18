//g++ pollete.cc -o pollete `pkg-config -cflags --libs opencv4`
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>

using namespace cv;
namespace fs = std::filesystem;

//Colorize
Mat colorize(const Mat& img, const Scalar& color) {
    Mat out = img.clone();
 
    for (int y = 0; y < img.rows; y++) {
        Vec3b* row_in = (Vec3b*)img.ptr<Vec3b>(y);
        Vec3b* row_out = out.ptr<Vec3b>(y);
        
        for (int x = 0; x < img.cols; x++) {
            //saturate_cast es una funcion de opencv que castea a unsigned char (pixel) sin que desborde de 255 (valor maxmimo)
            //para un color
            row_out[x][0] = saturate_cast<uchar>((row_in[x][0] + color[0]) / 2); 
            row_out[x][1] = saturate_cast<uchar>((row_in[x][1] + color[1]) / 2);
            row_out[x][2] = saturate_cast<uchar>((row_in[x][2] + color[2]) / 2);
        }
    }
    return out;
}

//Filtro Gaussiano 
Mat gaussianBlurManual(const Mat& img, int tamano_kernel = 5, double sigma = 1.0) {
    // Asegurar que el tamaño del kernel sea impar
    if (tamano_kernel % 2 == 0) {
        tamano_kernel++;
    }
    int radio = tamano_kernel / 2;
    Mat resultado = img.clone();
    //openCV genera el kernel gaussiano automáticamente
    Mat kernel_1d = getGaussianKernel(tamano_kernel, sigma);
    Mat kernel_2d = kernel_1d * kernel_1d.t();
    
    //Aplicar el kernel a cada píxel
    for (int y = radio; y < img.rows - radio; y++) {
        for (int x = radio; x < img.cols - radio; x++) {
            double suma_azul = 0, suma_verde = 0, suma_rojo = 0;
            // Multiplicar cada vecino por su peso en el kernel
            for (int i = -radio; i <= radio; i++) {
                for (int j = -radio; j <= radio; j++) {
                   //vec3b es un vector de 3 valores que representa un pixel (azul,verde,rojo)
                    Vec3b vecino = img.at<Vec3b>(y + i, x + j);
                    double peso = kernel_2d.at<double>(i + radio, j + radio);
                    suma_azul += vecino[0] * peso;
                    suma_verde += vecino[1] * peso;
                    suma_rojo += vecino[2] * peso;
                }
            }
            resultado.at<Vec3b>(y, x)[0] = (uchar)suma_azul;
            resultado.at<Vec3b>(y, x)[1] = (uchar)suma_verde;
            resultado.at<Vec3b>(y, x)[2] = (uchar)suma_rojo;
        }
    }
    return resultado;
}

//mediana 
Mat medianBlurManual(const Mat& img, int tamano_kernel = 3) {
    // Asegurar que el tamaño del kernel sea impar
    if (tamano_kernel % 2 == 0) {
        tamano_kernel++;
    }
    
    int radio = tamano_kernel / 2;
    Mat resultado = img.clone();
    
    for (int y = radio; y < img.rows - radio; y++) {
        for (int x = radio; x < img.cols - radio; x++) {
            std::vector<int> valores_azul, valores_verde, valores_rojo;
            //Recolectar todos los valores de la ventana
            for (int i = -radio; i <= radio; i++) {
                for (int j = -radio; j <= radio; j++) {
                    Vec3b vecino = img.at<Vec3b>(y + i, x + j);
                    valores_azul.push_back(vecino[0]);
                    valores_verde.push_back(vecino[1]);
                    valores_rojo.push_back(vecino[2]);
                }
            }
            //Ordenar las listas
            std::sort(valores_azul.begin(), valores_azul.end());
            std::sort(valores_verde.begin(), valores_verde.end());
            std::sort(valores_rojo.begin(), valores_rojo.end());       
            //Tomar el valor del medio (mediana)
            int posicion_media = valores_azul.size() / 2;
            resultado.at<Vec3b>(y, x)[0] = valores_azul[posicion_media];
            resultado.at<Vec3b>(y, x)[1] = valores_verde[posicion_media];
            resultado.at<Vec3b>(y, x)[2] = valores_rojo[posicion_media];
        }
    }
    return resultado;
}

//Detección de bordes Sobel 
Mat edgesSimple(const Mat& img) {
    Mat g;
    cvtColor(img, g, COLOR_BGR2GRAY);
    Mat e = Mat::zeros(g.size(), g.type());

    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    const int threshold = 100;
    for (int y = 1; y < g.rows - 1; y++) {
        uchar* e_row = e.ptr<uchar>(y);
        for (int x = 1; x < g.cols - 1; x++) {
            int sx = 0, sy = 0;
            for (int i = -1; i <= 1; i++) {
                const uchar* row = g.ptr<uchar>(y + i);
                for (int j = -1; j <= 1; j++) {
                    uchar v = row[x + j];
                    sx += gx[i + 1][j + 1] * v;
                    sy += gy[i + 1][j + 1] * v;
                }
            }
            int magnitude = static_cast<int>(sqrt(sx * sx + sy * sy));
            e_row[x] = (magnitude > threshold) ? 255 : 0;
        }
    }
    return e;
}

//Comic effect
Mat comicEffect(const Mat& img) {
    Mat edges = edgesSimple(img);
    Mat out;
    bilateralFilter(img, out, 5, 5, 75);
    
    for (int y = 0; y < img.rows; y++) {
        Vec3b* out_row = out.ptr<Vec3b>(y);
        const uchar* edge_row = edges.ptr<uchar>(y);
        
        for (int x = 0; x < img.cols; x++) {
            if (edge_row[x] > 200) {
                out_row[x] = Vec3b(0, 0, 0);
            }
        }
    }
    
    return out;
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_imagen>\n";
        return 1;
    }
    
    Mat img = imread(argv[1]);
    if (img.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen '" << argv[1] << "'\n";
        return -1;
    }
    
    std::cout << "Procesando imagen: " << argv[1] << "\n";
    std::cout << "Dimensiones: " << img.cols << "x" << img.rows << "\n\n";

    if (!fs::exists("resultados")) {
        fs::create_directory("resultados");
    }
    
    time_t inicio = clock();
    //Aplicar colorizaciones
    std::cout << "Aplicando efectos de color...\n";
    Mat red = colorize(img, {0, 0, 255});
    Mat green = colorize(img, {0, 255, 0});
    Mat blue = colorize(img, {255, 0, 0});
    Mat yellow = colorize(img, {0, 255, 255});
    
    //Aplicar filtros
    std::cout << "Aplicando filtros...\n";
    Mat red_b;
    bilateralFilter(red, red_b, 15, 200, 200);
    Mat green_b = gaussianBlurManual(green, 7, 2);
    Mat blue_b = medianBlurManual(blue, 7);
    Mat yellow_b = edgesSimple(yellow);
    cvtColor(yellow_b, yellow_b, COLOR_GRAY2BGR);
    
    //Combinar resultados
    std::cout << "Combinando resultados...\n";
    Mat top, bot, final_img;
    hconcat(red_b, green_b, top);
    hconcat(blue_b, yellow_b, bot);
    vconcat(top, bot, final_img);
    imwrite("resultados/combined_manual.png", final_img);
    
    //Efecto comic
    std::cout << "Generando efecto cómic...\n";
    Mat comic = comicEffect(final_img);
    imwrite("resultados/comic_manual.png", comic);
    time_t fin = clock();
    time_t total = fin - inicio;

    std::cout << "Tiempo total: " << (double) total / (double)CLOCKS_PER_SEC << " segundos" << std::endl;
    std::cout << "Resultados guardados en la carpeta 'resultados/'" << std::endl;
    
    return 0;
}