#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <random>

using namespace Eigen;

int main(int argc, char** argv)
{
    static std::default_random_engine e(time(0));
    // static std::default_random_engine e(0);  // fix rnadom seed
    static std::normal_distribution <double> n(0,1);

    double x = 3; // True slope
    VectorXd a = VectorXd::LinSpaced(17, -2, 2);
    VectorXd b = x*a + VectorXd::Zero(a.size()).unaryExpr([](double dummy){return n(e);});

    VectorXd xXa = x*a;

    matplot::figure();
    auto ax1 = matplot::gca();
    ax1->hold(matplot::on);
    ax1->plot(a, xXa, "k")->line_width(2)
                           .display_name("True Line");
    
    std::vector<double> A(a.size()), B(b.size());
    VectorXd::Map(A.data(), a.size()) = a;
    VectorXd::Map(B.data(), b.size()) = b;
    ax1->plot(a, b, "rx")->marker_size(10)
                          .display_name("Noisy Data");

    BDCSVD<MatrixXd> svd(a.matrix(), ComputeThinU | ComputeThinV);    
#if 0
    MatrixXd S = svd.singularValues().asDiagonal();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    VectorXd xtilde = V*S.inverse()*U.transpose()*b;
#else
    VectorXd xtilde = svd.solve(b);
#endif     
    
    VectorXd xtildeXa = xtilde(0)*a;
    std::cout << "x      = " << x << std::endl;
    std::cout << "xtilde = " << xtilde << std::endl;

    ax1->plot(a, xtildeXa, "b--")->line_width(4)
                                  .display_name("Regression Line");
    ax1->xlabel("a");
    ax1->ylabel("b");
    ax1->grid(true);
    ax1->grid_line_style(matplot::line_spec("k--"));
    matplot::legend()->location(matplot::legend::general_alignment::topleft);
    matplot::show();

    return 0;
}