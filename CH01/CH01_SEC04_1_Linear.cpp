#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <random>

namespace plt = matplotlibcpp;
using namespace Eigen;


int main(int argc, char** argv)
{
    static std::default_random_engine e(time(0));
    // static std::default_random_engine e(0);  // fix rnadom seed
    static std::normal_distribution <float> n(0,1);

    float x = 3; // True slope
    VectorXf a = VectorXf::LinSpaced(17, -2, 2);
    VectorXf b = x*a + VectorXf::Zero(a.size()).unaryExpr([](float dummy){return n(e);});

    VectorXf xXa = x*a;

    std::map<std::string, std::string> kwargs;

    kwargs["color"] = "k";
    kwargs["linewidth"] = "2";
    kwargs["label"] = "True line";
    plt::plot(a, xXa, kwargs);
    
    kwargs.erase("linewidth");
    kwargs["color"] = "r";
    kwargs["markersize"] = "10";
    kwargs["label"] = "Noisy data";
    plt::plot(a, b, "x", kwargs);
    
    BDCSVD<MatrixXf> svd(a.matrix(), ComputeThinU | ComputeThinV);    
#if 0
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();
    VectorXf xtilde = V*S.inverse()*U.transpose()*b;
#else
    VectorXf xtilde = svd.solve(b);
#endif     
    
    VectorXf xtildeXa = xtilde(0)*a;
    std::cout << "x      = " << x << std::endl;
    std::cout << "xtilde = " << xtilde << std::endl;

    kwargs.erase("marker");
    kwargs.erase("markersize");
    kwargs["color"] = "b";
    kwargs["linewidth"] = "4";
    kwargs["label"] = "Regression line";
    plt::plot(a, xtildeXa, "--", kwargs);

    plt::xlabel("a");
    plt::ylabel("b");
    plt::grid();
    plt::legend();
    plt::show();
    
    return 0;
}