#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <random>

namespace plt = matplotlibcpp;
using namespace Eigen;

typedef Matrix<float, -1, -1, RowMajor> MatrixXfR; // PCA Literature uses Row Major Matrices

int main(int argc, char** argv)
{
    Vector2f xC, sig;
    xC << 2, 1;
    sig << 2, 0.5;

    float theta = M_PI/3;

    Matrix2f R;
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);

    int nPoints = 10000;

    static std::default_random_engine e(time(0));
    static std::normal_distribution <float> n(0,1);
    MatrixXfR X = R * sig.asDiagonal() * MatrixXfR::Zero(2,nPoints).unaryExpr([](float dummy){return n(e);})
                  + xC.asDiagonal() * MatrixXfR::Ones(2,nPoints);
    
    Vector2f Xavg = X.rowwise().mean();
    MatrixXfR B = X;
    B.row(0) = B.row(0).array() - Xavg(0);
    B.row(1) = B.row(1).array() - Xavg(1);

    B = B/std::sqrt(nPoints);
    BDCSVD<MatrixXf> svd(B, ComputeThinU);
    MatrixXf S = svd.singularValues().asDiagonal();
    MatrixXf U = svd.matrixU();

    VectorXf theta_v = VectorXf::LinSpaced(100, 0, 2*M_PI);
    MatrixXfR UnitCircle(2,theta_v.size());
    UnitCircle.row(0) = theta_v.array().cos();
    UnitCircle.row(1) = theta_v.array().sin();

    MatrixXfR Xstd = U * S * UnitCircle;

    MatrixXfR Xsig1 = Xstd + Xavg.asDiagonal()*MatrixXfR::Ones(2,theta_v.size());
    MatrixXfR Xsig2 = 2*Xstd + Xavg.asDiagonal()*MatrixXfR::Ones(2,theta_v.size());
    MatrixXfR Xsig3 = 3*Xstd + Xavg.asDiagonal()*MatrixXfR::Ones(2,theta_v.size());

    plt::plot(X.row(0), X.row(1), "k.");
    plt::plot(Xsig1.row(0), Xsig1.row(1), "r-", {{"linewidth", "3"}});
    plt::plot(Xsig2.row(0), Xsig2.row(1), "r-", {{"linewidth", "3"}});
    plt::plot(Xsig3.row(0), Xsig3.row(1), "r-", {{"linewidth", "3"}});
    plt::plot(std::vector{Xavg(0), Xavg(0)+U(0,0)*S(0,0)}, std::vector{Xavg(1), Xavg(1)+U(1,0)*S(0,0)}, {{"linewidth", "3"}, {"color", "cyan"}});
    plt::plot(std::vector{Xavg(0), Xavg(0)+U(0,1)*S(1,1)}, std::vector{Xavg(1), Xavg(1)+U(1,1)*S(1,1)}, {{"linewidth", "3"}, {"color", "cyan"}});
    plt::grid();
    plt::xlim(-6,8);
    plt::ylim(-6,8);
    plt::show();
}
