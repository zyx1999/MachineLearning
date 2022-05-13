#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
    // Eigen::MatrixXd m(4,4);
    // m(0,0) = 1;
    // m(2,1) = 3;
    // cout<<m<<endl;
    // Eigen::Vector4d a = m.col(1);
    // cout<<a<<endl;
    // cout<<m.col(0)<<endl;
    ifstream inFile("../../dataset/iris.csv", ios::in);
    string lineStr;
    int cols = 0;
    vector<vector<double>> data;
    getline(inFile, lineStr);
    cout << lineStr << endl;
    Eigen::Matrix4Xd attrs(4, 150);
    Eigen::VectorXi species(150);
    string species_1 = "setosa";
    string species_2 = "versicolor";
    string species_3 = "virginica";

    int i = 0;
    while (getline(inFile, lineStr)) {
        // cout<<lineStr<<endl;
        string field;
        istringstream sin(lineStr);

        // index
        getline(sin, field, ',');

        // sepal.length
        getline(sin, field, ',');
        attrs(0, i) = stod(field);

        // sepal.width
        getline(sin, field, ',');
        attrs(1, i) = stod(field);

        // petal.length
        getline(sin, field, ',');
        attrs(2, i) = stod(field);

        // petal.width
        getline(sin, field, ',');
        attrs(3, i) = stod(field);

        // species
        getline(sin, field, ',');
        if (field.compare(1, 6, species_1) == 0) {
            species(i) = -1;
        // } else if (field.compare(1, 10, species_2) == 0) {
        //     species(i) = 1;
        } else if (field.compare(1, 9, species_3) == 0) {
            species(i) = 1;
        }
        i++;
    }
    cout<<species.transpose()<<endl;
    return 0;
}
