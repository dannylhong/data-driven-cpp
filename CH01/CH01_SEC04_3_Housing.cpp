#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <math.h>
#include <random>
#include <fstream>

namespace plt = matplotlibcpp;
using namespace Eigen;

template<typename M>
M load_data (const std::string & path);

namespace Eigen {
    template<class T>
    void swap(T&& a, T&& b){
        a.swap(b);
    }
}

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/housing.data";
    MatrixXf H = load_data<MatrixXf>(path);

    MatrixXf b = H.block(0, H.cols() - 1, H.rows(), 1);
    MatrixXf A = H;
    A.block(0, H.cols() - 1, H.rows(), 1) = VectorXf::Ones(H.rows());

    BDCSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
#if 0
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();
    VectorXf x = V*S.inverse()*U.transpose()*b;
#else
    VectorXf x = svd.solve(b);
#endif     

    
    MatrixXf btilde = A*x;
    
    std::map<std::string, std::string> kwargs;

    kwargs["color"] = "k";
    kwargs["linewidth"] = "2";
    kwargs["label"] = "Housing Value";
    plt::plot(b, kwargs);
    
    kwargs["color"] = "r";
    kwargs["linewidth"] = "1.5";
    kwargs["marker"] = "o";
    kwargs["markersize"] = "6";
    kwargs["label"] = "Regression";
    plt::plot(btilde, kwargs);
    plt::xlabel("Neighborhood");
    plt::ylabel("Median Home Value [$1k]");
    plt::legend();
    plt::show();


    MatrixXf bA(A.rows(), A.cols()+1) ;
    bA.block(0, 0, A.rows(), 1) = b;
    bA.block(0, 1, A.rows(), A.cols()) = A;

    std::sort(bA.rowwise().begin(), bA.rowwise().end(),
     [](auto const& r1, auto const& r2){return r1(0)<r2(0);});

    kwargs.erase("marker");
    kwargs.erase("markersize");
    kwargs["color"] = "k";
    kwargs["linewidth"] = "2";
    kwargs["label"] = "Housing Value";
    plt::plot(bA.block(0, 0, A.rows(), 1), kwargs);

    btilde = bA.block(0, 1, A.rows(), A.cols())*x;
    kwargs["color"] = "r";
    kwargs["linewidth"] = "1.5";
    kwargs["marker"] = "o";
    kwargs["markersize"] = "6";
    kwargs["label"] = "Regression";
    plt::plot(btilde, kwargs);
    plt::xlabel("Neighborhood");
    plt::legend();

    plt::show();

    return 0;
}

template<typename M>
M load_data (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            if(cell != "")
                values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}
