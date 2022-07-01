#include <matioCpp/matioCpp.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

#define TIME_CHECK 1

int main(int argc,char **argv)
{
    matioCpp::File input("../../../DATA/allFaces.mat");
    
    matioCpp::MultiDimensionalArray<double> faces = input.read("faces").asMultiDimensionalArray<double>(); //Read a Cell Array named "cell_array"
    matioCpp::Element<double> m = input.read("m").asElement<double>();
    matioCpp::Element<double> n = input.read("n").asElement<double>();
    matioCpp::Vector<double> nfaces = input.read("nfaces").asVector<double>();

    Eigen::VectorXd cumsum = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(nfaces.data()-1, nfaces.size()+1);
    cumsum(0) = 0;
    for(int i = 1; i<cumsum.size(); i++)
        cumsum(i) += cumsum(i-1);    
    Eigen::MatrixXd facesMat = matioCpp::to_eigen(faces); 

    // We use the first 36 people for training data
    Eigen::MatrixXd trainingFaces = facesMat.block(0,0,facesMat.rows(),cumsum(36));
    Eigen::VectorXd avgFace = facesMat.rowwise().mean();

    // Compute eigenfaces on mean-subtracted training data
    Eigen::MatrixXd X = trainingFaces;
    X.colwise() -= avgFace;
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

#if TIME_CHECK

    std::cout << "SVD computation started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

#endif

    Eigen::MatrixXd S = svd.singularValues().asDiagonal();
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

#if TIME_CHECK
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "Elapsed time for SVD is " << float_ms.count() << " milliseconds" << std::endl;
#endif

    plt::imshow(avgFace.reshaped(n(), m()), {{"cmap", "gray"}});
    plt::axis("off");
    plt::show();

    plt::imshow(U.col(0).reshaped(n(), m()), {{"cmap", "gray"}});
    plt::axis("off");
    plt::show();

#if 1
    // Now show eigenface reconstruction of image that was omitted from test set
    Eigen::MatrixXd testFace = facesMat.col(cumsum(36));
#else
    // Reconstruction of Dog from Eigen Faces
    std::string path = "../../../DATA/dog.jpg";
    cv::Mat A = cv::imread(path, cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd Dog, DogCrop;
    cv::cv2eigen(A, Dog);
    DogCrop = Dog.block(.7*n(),20,(5.6*n()),(5.6*m()));
    cv::eigen2cv(DogCrop, A);
    cv::resize(A, A, cv::Size(m(),n()));
    cv::cv2eigen(A, DogCrop);
    Eigen::MatrixXd testFace = DogCrop.reshaped();
#endif
    plt::imshow(testFace.reshaped(n(), m()), {{"cmap", "gray"}});
    plt::title("Original Image");
    plt::axis("off");
    plt::show();

    Eigen::MatrixXd testFaceMS = testFace - avgFace;
    std::vector<int> r_list = {25, 50, 100, 200, 400, 800, 1600};

    for( auto r : r_list ){
        Eigen::MatrixXd reconFace = avgFace + U.block(0,0,U.rows(),r)*U.block(0,0,U.rows(),r).transpose()*testFaceMS;
        plt::imshow(reconFace.reshaped(n(), m()), {{"cmap", "gray"}});
        plt::title("r = " + std::to_string(r));
        plt::axis("off");
        plt::show();
    }

    return 0;
}

