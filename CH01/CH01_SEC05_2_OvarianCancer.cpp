#include <Eigen/Dense>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <fstream>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path);
std::vector<std::string> load_txt(const std::string & path);

int main(int argc, char** argv)
{
    std::string path = "../../../DATA/ovariancancer_obs.csv";
    MatrixXf obs = load_csv<MatrixXf>(path);
    path = "../../../DATA/ovariancancer_grp.csv";
    std::vector<std::string> grp = load_txt(path);

    BDCSVD<MatrixXf> svd(obs, ComputeThinU | ComputeThinV);
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();

    matplot::figure()->size(1000, 500);
    auto ax1 = matplot::subplot(1,2,0);
    ax1->semilogy(svd.singularValues(), "ko-")->marker_face(true);
    ax1->ylim({0.5, 1e3});
    ax1->ytickformat("%.0e");
        
    // plt::semilogy(svd.singularValues());
    // plt::show();

    VectorXf cumsum = svd.singularValues();
    for(int i=1; i<cumsum.size(); i++)
        cumsum(i) += cumsum(i-1);    
    cumsum = cumsum/svd.singularValues().sum();
    
    auto ax2 = matplot::subplot(1,2,1);
    ax2->plot(cumsum, "ko-")->marker_face(true);
    ax2->ylim({0.5, 1.1});
    matplot::show();

    // plt::plot(cumsum);
    // plt::show();

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
            if(cell != ""){
                values.push_back(std::stof(cell));
            }
                
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}


std::vector<std::string> load_txt(const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<std::string> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, '.')) {
            if(cell != ""){
                values.push_back(cell);
            }
                
        }
        ++rows;
    }
    return values;
}