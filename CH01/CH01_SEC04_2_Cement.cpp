#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <math.h>
#include <random>
#include <fstream>

namespace plt = matplotlibcpp;
using namespace Eigen;

template<typename M>
M load_csv (const std::string & path);

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/hald_ingredients.csv";
    MatrixXf A = load_csv<MatrixXf>(path);
    path = "../../../DATA/hald_heat.csv";
    MatrixXf b = load_csv<MatrixXf>(path);


    BDCSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
#if 0 // method 1
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();
    VectorXf x = V*S.inverse()*U.transpose()*b;
#else // method 2
    VectorXf x = svd.solve(b);
#endif     

    MatrixXf btilde = A*x;
    
    std::map<std::string, std::string> kwargs;

    kwargs["color"] = "k";
    kwargs["linewidth"] = "2";
    kwargs["label"] = "Hear Data";
    plt::plot(b, kwargs);
    
    kwargs["color"] = "r";
    kwargs["linewidth"] = "1.5";
    kwargs["marker"] = "o";
    kwargs["markersize"] = "6";
    kwargs["label"] = "Regression";
    plt::plot(btilde, kwargs);
    plt::legend();
    plt::show();

    return 0;
}

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}