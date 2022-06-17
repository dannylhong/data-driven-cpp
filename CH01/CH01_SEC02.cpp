#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <matplotlibcpp.h>
#include <matplot/matplot.h>

#define TIME_CHECK 0

#if TIME_CHECK
#include <chrono>
#endif

namespace plt = matplotlibcpp;
using namespace Eigen;

typedef Matrix<uchar, -1, -1> MatrixXc;
typedef Matrix<uchar, -1, 1> VectorXc;

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/dog.jpg";
    cv::Mat A = cv::imread(path, cv::IMREAD_GRAYSCALE);
    MatrixXd X;
    cv::cv2eigen(A, X);

    // // matplot::imshow is much slower than plt::imshow
    // MatrixXc Xc = X.cast<uchar>();

    // // matplot::imshow uses vector of vectors as input - copy Eigen Matrix into std::vector<std::vector<>> using Map
    // std::vector<std::vector<uchar>> C(Xc.rows());
    // for ( int i = 0 ; i < Xc.rows() ; i++ ){
    //     C[i].resize(Xc.cols());
    //     VectorXc::Map(C[i].data(), Xc.cols()) = Xc.row(i);
    // }
    // matplot::imshow(C);
    // matplot::title("original image");
    // matplot::show();

    plt::figure();
    plt::imshow(X, {{"cmap", "gray"}});
    plt::axis("off");
    plt::title("original image");
    plt::show();

#if TIME_CHECK

    std::cout << "SVD computation started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
#endif

    BDCSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);

#if TIME_CHECK
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "Elapsed time for main algorithm is " << float_ms.count() << " milliseconds" << std::endl;
#endif
    MatrixXd S = svd.singularValues().asDiagonal();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    std::vector<int> range = {5, 20, 100};

    MatrixXd Xapprox[range.size()];

    int i = 0;
    for (auto r : range)
    {
        Xapprox[i] = U.block(0,0,U.rows(),r)*S.block(0,0,r,r)*V.block(0,0,V.rows(),r).transpose(); 
        
        // Xc = Xapprox[i].cast<uchar>();
        // for ( int i = 0 ; i < Xc.rows() ; i++ ){
        //     VectorXc::Map(C[i].data(), Xc.cols()) = Xc.row(i);
        // }

        // matplot::imshow(C);
        // matplot::title("original image");
        // matplot::show();

        plt::figure();
        plt::imshow(Xapprox[i], {{"cmap", "gray"}});
        plt::axis("off");
        plt::title("r = " + std::to_string(r));
        plt::show();
    }

    matplot::figure();
    auto ax1 = matplot::gca();
    ax1->semilogy(svd.singularValues())->line_width(2);
    ax1->title("Singular Values");
    ax1->ytickformat("%.0e");
    ax1->xlim({-75, 1600});
    ax1->grid(true);
    matplot::show();

    // plt::figure();
    // plt::semilogy(svd.singularValues());
    // plt::title("Singular Values");
    // plt::grid();
    // plt::show();

    VectorXd cumsum = svd.singularValues();
    for(int i=1; i<cumsum.size(); i++)
        cumsum(i) += cumsum(i-1);    
    cumsum = cumsum/svd.singularValues().sum();

    matplot::figure();
    auto ax2 = matplot::gca();
    ax2->plot(cumsum)->line_width(2);
    ax2->title("Singular Values: Cumulative Sum");
    ax2->xlim({-75, 1600});
    ax2->grid(true);
    matplot::show();

    // plt::figure();
    // plt::plot(cumsum);
    // plt::title("Singular Values: Cumulative Sum");
    // plt::grid();
    // plt::show();

    return 0;
}
