#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <fstream>

namespace plt = matplotlibcpp;
using namespace Eigen;

template<typename M>
M load_data (const std::string & path);

// For Sorting Matrix by row according to the first column elements
namespace Eigen {
    template<class T>
    void swap(T&& a, T&& b){
        a.swap(b);
    }
}

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/housing.data";
    MatrixXd H = load_data<MatrixXd>(path);

    MatrixXd b = H.col(H.cols() - 1);
    MatrixXd A = H;
    A.col(H.cols() - 1) = VectorXd::Ones(H.rows());

    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd x = svd.solve(b);
    
    MatrixXd btilde = A*x;
    
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


    MatrixXd bA(A.rows(), A.cols()+1) ;
    bA.col(0) = b;
    bA.block(0, 1, A.rows(), A.cols()) = A;

    std::sort(bA.rowwise().begin(), bA.rowwise().end(),
              [](auto const& r1, auto const& r2){return r1(0)<r2(0);});

    kwargs.erase("marker");
    kwargs.erase("markersize");
    kwargs["color"] = "k";
    kwargs["linewidth"] = "2";
    kwargs["label"] = "Housing Value";
    plt::plot(bA.col(0), kwargs);

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


    VectorXd A_mean = A.colwise().mean();

    MatrixXd A2 = A - MatrixXd::Ones(A.rows(),1)*A_mean.transpose();

    double A2std;
    for (int j = 0; j < A.cols() - 1; j++){
        A2std = std::sqrt(A2.col(j).cwiseAbs2().sum()/(A2.rows()));
        A2.col(j) = A2.col(j)/A2std;
    }
    A2.col(A2.cols() - 1) = VectorXd::Ones(A2.rows());

    // std::ofstream file("A2.txt");
    // if (file.is_open())
    // {
    //     file << A2;
    // }

    BDCSVD<MatrixXd> svd2(A2, ComputeThinU | ComputeThinV);
    MatrixXd S = svd2.singularValues().asDiagonal();
    MatrixXd U = svd2.matrixU();
    MatrixXd V = svd2.matrixV();
    x = V*S.inverse()*U.transpose()*b;

    std::cout << x; // I have no idea why the result differs from the python one. If I compare A2 matrix, maximum difference is about 5e-6.

    std::vector<double> vec(x.data(), x.data() + x.size() - 1);
    plt::bar(vec);
    plt::show();

    return 0;
}

template<typename M>
M load_data (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
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