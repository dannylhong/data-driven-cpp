#include <Eigen/Dense>
#include <matplot/matplot.h>

using namespace Eigen;

int main(int argc, char** argv)
{
    double theta[3] = {M_PI/15, -M_PI/9, -M_PI/20};
    DiagonalMatrix<double, 3> Sigma(3, 1, 0.5);

    // Rotation about x axis
    Matrix3d Rx, Ry, Rz;
    Rx << 1,                  0,                   0,
          0, std::cos(theta[0]), -std::sin(theta[0]),
          0, std::sin(theta[0]),  std::cos(theta[0]);

    Ry <<  std::cos(theta[1]), 0, std::sin(theta[1]),
                            0, 1,                  0,
          -std::sin(theta[1]), 0, std::cos(theta[1]);

    Rz << std::cos(theta[2]), -std::sin(theta[2]), 0,
          std::sin(theta[2]),  std::cos(theta[2]), 0,
                           0,                   0, 1;

    Matrix3d X = Rz*Ry*Rx*Sigma;

    VectorXd u = VectorXd::LinSpaced(100, -M_PI, M_PI);
    VectorXd v = VectorXd::LinSpaced(100,     0, M_PI);
    MatrixXd x = u.array().cos().matrix()*v.array().sin().matrix().transpose();
    MatrixXd y = u.array().sin().matrix()*v.array().sin().matrix().transpose();
    MatrixXd z = VectorXd::Ones(u.size())*v.array().cos().matrix().transpose();

    std::vector<std::vector<double>> XX(x.rows()), YY(x.rows()), ZZ(x.rows());
    for ( int i = 0 ; i < x.rows() ; i++ ){
        XX[i].resize(x.cols());
        YY[i].resize(x.cols());
        ZZ[i].resize(x.cols());
        VectorXd::Map(XX[i].data(), x.cols()) = x.row(i);
        VectorXd::Map(YY[i].data(), x.cols()) = y.row(i);
        VectorXd::Map(ZZ[i].data(), x.cols()) = z.row(i);
    }
    matplot::figure();
    auto ax1 = matplot::gca();
    ax1->surf(XX,YY,ZZ)->face_alpha(0.6).line_width(0.1).edge_color("none");
    ax1->colormap(matplot::palette::jet());
    ax1->xlim({-3, 3});
    ax1->ylim({-3, 3});
    ax1->zlim({-3, 3});
    matplot::show();

    MatrixXd xR = MatrixXd::Zero(x.rows(), x.cols());
    MatrixXd yR = MatrixXd::Zero(y.rows(), y.cols());
    MatrixXd zR = MatrixXd::Zero(z.rows(), z.cols());

    Vector3d vec, vecR;
    for(int i=0; i<x.rows(); i++){
          for(int j=0; j<x.cols(); j++){
                vec << x(i,j), y(i,j), z(i,j) ;
                vecR = X * vec;
                xR(i,j) = vecR(0);
                yR(i,j) = vecR(1);
                zR(i,j) = vecR(2);
          }
    }
    for ( int i = 0 ; i < x.rows() ; i++ ){
        VectorXd::Map(XX[i].data(), x.cols()) = xR.row(i);
        VectorXd::Map(YY[i].data(), x.cols()) = yR.row(i);
        VectorXd::Map(ZZ[i].data(), x.cols()) = zR.row(i);
    }
    matplot::figure();
    auto ax2 = matplot::gca();
    ax2->surf(XX,YY,ZZ)->face_alpha(0.6).line_width(0.1).edge_color("none");
    ax2->colormap(matplot::palette::jet());
    ax2->xlim({-3, 3});
    ax2->ylim({-3, 3});
    ax2->zlim({-3, 3});
    matplot::show();
    
    return 0;
}