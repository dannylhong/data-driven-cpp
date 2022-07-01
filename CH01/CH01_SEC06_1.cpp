#include <matioCpp/matioCpp.h>
#include <Eigen/Dense>
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

int main(int argc,char **argv)
{
    matioCpp::File input("../../../DATA/allFaces.mat");
    
    matioCpp::MultiDimensionalArray<double> faces = input.read("faces").asMultiDimensionalArray<double>(); //Read a Cell Array named "cell_array"
    matioCpp::Element<double> m = input.read("m").asElement<double>();
    matioCpp::Element<double> n = input.read("n").asElement<double>();
    matioCpp::Vector<double> nfaces = input.read("nfaces").asVector<double>();
    Eigen::VectorXd cumsum = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(nfaces.data()-1, nfaces.size());
    cumsum(0) = 0;
    for(int i = 1; i<cumsum.size(); i++)
        cumsum(i) += cumsum(i-1);    

    Eigen::MatrixXd facesMat = matioCpp::to_eigen(faces);     

    Eigen::MatrixXd allPersons = Eigen::MatrixXd::Zero(6*n(), 6*m());
    
    int count = 0;
    for(int j = 0; j < 6; ++j){
        for(int k = 0; k < 6; ++k){
            allPersons.block(j*n(), k*m(), n(), m()) = facesMat.col(cumsum(count)).reshaped(n(), m());
            count++;
        }
    }

    plt::imshow(allPersons, {{"cmap", "gray"}});
    plt::axis("off");
    plt::show();

    for(int person = 0; person < nfaces.size(); ++person){
        Eigen::MatrixXd subset = facesMat.block(0, cumsum(person), facesMat.rows(), nfaces(person));
        Eigen::MatrixXd allFaces = Eigen::MatrixXd::Zero(8*n(), 8*m());
        count = 0;
        for(int j = 0; j < 8; ++j){
            for(int k = 0; k < 8; ++k){
                if(count < nfaces(person)){
                    allFaces.block(j*n(), k*m(), n(), m()) = subset.col(count).reshaped(n(), m());
                    count++;
                }
            }
        }
        plt::imshow(allFaces, {{"cmap", "gray"}});
        plt::axis("off");
        plt::show();
    }


    return 0;
}

