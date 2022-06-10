#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <matplotlibcpp.h>

#define TIME_CHECK 0

#if TIME_CHECK
#include <chrono>
#endif

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/dog.jpg";
    cv::Mat A = cv::imread(path, cv::IMREAD_GRAYSCALE);
    MatrixXf X;
    cv::cv2eigen(A, X);

    plt::figure();
    plt::imshow(X, {{"cmap", "gray"}});
    plt::axis("off");
    plt::title("original image");
    plt::show();

#if TIME_CHECK
    auto start = std::chrono::high_resolution_clock::now();
#endif

    BDCSVD<MatrixXf> svd(X, ComputeThinU | ComputeThinV);

#if TIME_CHECK
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "Elapsed time for main algorithm is " << float_ms.count() << " milliseconds" << std::endl;
#endif
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();

    std::vector<int> range = {5, 20, 100};

    MatrixXf Xapprox[range.size()];

    int i = 0;
    for (auto r : range)
    {
        Xapprox[i] = U.block(0,0,U.rows(),r)*S.block(0,0,r,r)*V.block(0,0,V.rows(),r).transpose(); 
        plt::figure();
        plt::imshow(Xapprox[i], {{"cmap", "gray"}});
        plt::axis("off");
        plt::title("r = " + std::to_string(r));
        plt::show();
    }

    plt::figure();
    plt::semilogy(svd.singularValues());
    plt::title("Singular Values");
    plt::grid();
    plt::show();

    VectorXf cumsum = svd.singularValues();
    for(int i=1; i<cumsum.size(); i++)
        cumsum(i) += cumsum(i-1);    
    cumsum = cumsum/svd.singularValues().sum();

    plt::figure();
    plt::plot(cumsum);
    plt::title("Singular Values: Cumulative Sum");
    plt::grid();
    plt::show();

    return 0;
}
