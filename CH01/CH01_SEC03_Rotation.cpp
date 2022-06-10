#include <Eigen/Dense>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(int argc, char** argv)
{
    float theta[3] = {M_PI/15, -M_PI/9, -M_PI/20};
    DiagonalMatrix<float, 3> Sigma(3, 1, 0.5);

    // Rotation about x axis
    Matrix3f Rx, Ry, Rz;
    Rx << 1,                  0,                   0,
          0, std::cos(theta[0]), -std::sin(theta[0]),
          0, std::sin(theta[0]),  std::cos(theta[0]);

    Ry <<  std::cos(theta[1]), 0, std::sin(theta[1]),
                            0, 1,                  0,
          -std::sin(theta[1]), 0, std::cos(theta[1]);

    Rz << std::cos(theta[2]), -std::sin(theta[2]), 0,
          std::sin(theta[2]),  std::cos(theta[2]), 0,
                           0,                   0, 1;

    Matrix3f X = Rz*Ry*Rx*Sigma;

    VectorXf u = VectorXf::LinSpaced(100, -M_PI, M_PI);
    VectorXf v = VectorXf::LinSpaced(100,     0, M_PI);
    MatrixXf x = u.array().cos().matrix()*v.array().sin().matrix().transpose();
    MatrixXf y = u.array().sin().matrix()*v.array().sin().matrix().transpose();
    MatrixXf z = VectorXf::Ones(u.size())*v.array().cos().matrix().transpose();

    plt::plot_surface(x,y,z, {{"cmap", "jet"}});
    plt::show();

    MatrixXf xR = MatrixXf::Zero(x.rows(), x.cols());
    MatrixXf yR = MatrixXf::Zero(y.rows(), y.cols());
    MatrixXf zR = MatrixXf::Zero(z.rows(), z.cols());

    Vector3f vec, vecR;
    for(int i=0; i<x.rows(); i++){
          for(int j=0; j<x.cols(); j++){
                vec << x(i,j), y(i,j), z(i,j) ;
                vecR = X * vec;
                xR(i,j) = vecR(0);
                yR(i,j) = vecR(1);
                zR(i,j) = vecR(2);
          }
    }

    plt::plot_surface(xR,yR,zR, {{"cmap", "jet"}});
    plt::show();
    
    return 0;
}