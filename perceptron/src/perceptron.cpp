#include <Eigen/Core>
#include <Eigen/Dense>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

class perceptron {
   public:
    perceptron() { w.setZero(); };
    ~perceptron(){};
    void readData(Eigen::Matrix2Xd&, Eigen::VectorXd&);
    void readIris(Eigen::Matrix2Xd&, Eigen::VectorXd&);
    double dot(vector<double>, vector<double>);
    double sign(vector<double>, vector<double>, double);
    void fit(Eigen::Matrix2Xd, Eigen::VectorXd);
    void fit_duality(Eigen::MatrixXd, Eigen::Matrix2Xd, Eigen::VectorXd);
    vector<pair<Eigen::Vector3d, int>> fit_voted(int, Eigen::Matrix2Xd,
                                                 Eigen::VectorXd);
    void predict_voted(vector<pair<Eigen::Vector3d, int>>, Eigen::Vector2d);
    void getGramMatrix(Eigen::Matrix2Xd, Eigen::MatrixXd&);
    inline Eigen::Vector2d getWeight() { return w; }
    inline double getBias() { return bias; }
    inline void clear() {
        bias = 0;
        w.setZero();
    }

   private:
    double eta = 1;
    double bias = 0;
    Eigen::Vector2d w;
};

void perceptron::getGramMatrix(Eigen::Matrix2Xd x_train,
                               Eigen::MatrixXd& gram) {
    gram = x_train.transpose() * x_train;
}

void perceptron::readIris(Eigen::Matrix2Xd& attrs, Eigen::VectorXd& species) {
    ifstream inFile("../../dataset/iris.csv", ios::in);
    string lineStr;
    int cols = 0;
    vector<vector<double>> data;
    getline(inFile, lineStr);
    // cout << lineStr << endl;
    // Eigen::Matrix4Xd attrs(4, 100);
    // Eigen::VectorXi species(100);
    string species_1 = "setosa";
    string species_2 = "versicolor";
    string species_3 = "virginica";
    int i = 0;
    while (getline(inFile, lineStr)) {
        string field;
        istringstream sin(lineStr);

        // index
        getline(sin, field, ',');

        // sepal.length
        getline(sin, field, ',');
        // attrs(0, i) = stod(field);

        // sepal.width
        getline(sin, field, ',');
        // attrs(1, i) = stod(field);

        // petal.length
        getline(sin, field, ',');
        double petal_len = stod(field);
        // attrs(0, i) = stod(field);

        // petal.width
        getline(sin, field, ',');
        double petal_wid = stod(field);
        // attrs(1, i) = stod(field);

        // species
        getline(sin, field, ',');
        if (field.compare(1, 6, species_1) == 0) {
            species(i) = -1;
            attrs(0, i) = petal_len;
            attrs(1, i) = petal_wid;
            i++;
        } else if (field.compare(1, 10, species_2) == 0) {
            species(i) = 0;
        } else if (field.compare(1, 9, species_3) == 0) {
            species(i) = 1;
            attrs(0, i) = petal_len;
            attrs(1, i) = petal_wid;
            i++;
        }
    }
}

void perceptron::readData(Eigen::Matrix2Xd& x_train, Eigen::VectorXd& y_train) {
    ifstream inFile("../../dataset/separable_blobs.csv", ios::in);
    string lineStr;
    // get header
    getline(inFile, lineStr);
    int i = 0;
    while (getline(inFile, lineStr)) {
        stringstream ss(lineStr);
        string str;
        getline(ss, str, ',');
        x_train(0, i) = stod(str);

        getline(ss, str, ',');
        x_train(1, i) = stod(str);

        getline(ss, str, ',');
        double y = stod(str);
        y_train(i) = y > 0 ? 1 : -1;
        i++;
    }
}

double perceptron::dot(vector<double> x, vector<double> w) {
    double dot = 0;
    for (int i = 0; i < x.size(); i++) {
        dot += w[i] * x[i];
    }
    return dot;
}

double perceptron::sign(vector<double> x, vector<double> w, double bias) {
    double wtx = dot(x, w);
    wtx += bias;
    return wtx;
}

void perceptron::fit(Eigen::Matrix2Xd x_train, Eigen::VectorXd y_train) {
    bool is_wrong = false;
    int updates = 0;
    while (!is_wrong) {
        int wrong_count = 0;
        for (int i = 0; i < x_train.cols(); i++) {
            Eigen::Vector2d x_i = x_train.col(i);
            double y_i = y_train(i);
            if (y_i * (w.transpose() * x_i + bias) <= 0) {
                w += eta * y_i * x_i;
                bias += eta * y_i;
                wrong_count += 1;
            }
        }
        updates++;
        if (wrong_count == 0) {
            is_wrong = true;
        }
    }
    cout << "updates: " << updates << endl;
    cout << "Original Form" << endl;
}

void perceptron::fit_duality(Eigen::MatrixXd gram, Eigen::Matrix2Xd x_train,
                             Eigen::VectorXd y_train) {
    // vector<double> alpha(gram.size(), 0);
    int N = x_train.cols();
    Eigen::VectorXd alpha(N);
    alpha.setZero();
    Eigen::VectorXd ay(N);
    ay.setZero();
    bool is_wrong = false;
    int updates = 0;
    while (!is_wrong) {
        int wrong_count = 0;
        for (int i = 0; i < N; i++) {
            Eigen::Vector2d x_i = x_train.col(i);
            double y_i = y_train(i);
            if (y_i * (x_i.dot(x_train * ay) + bias) <= 0) {
                alpha(i) += eta;
                bias += eta * y_i;
                ay(i) = alpha(i) * y_train(i);
                wrong_count++;
            }
        }
        updates++;
        if (wrong_count == 0) {
            is_wrong = true;
        }
    }
    for (int i = 0; i < alpha.rows(); i++) {
        // cout << alpha(i) << " ";
        w += alpha(i) * y_train(i) * x_train.col(i);
    }
    cout << endl;
    cout << "updates: " << updates << endl;
    cout << "Duality Form" << endl;
}

vector<pair<Eigen::Vector3d, int>> perceptron::fit_voted(
    int T, Eigen::Matrix2Xd x_train, Eigen::VectorXd y_train) {
    int M = x_train.cols();
    Eigen::Matrix3Xd x_homo(3, M);
    x_homo.topLeftCorner(2, M) = x_train;
    x_homo.row(2).fill(1);

    int k = 0;
    vector<pair<Eigen::Vector3d, int>> voted;
    Eigen::Vector3d v_1(0, 0, 0);
    int c_1 = 0;
    voted.push_back({v_1, c_1});

    while (T-- > 0) {
        for (int i = 0; i < M; i++) {
            Eigen::Vector3d x_i = x_homo.col(i);
            double y_i = y_train(i);
            double y_hat = x_i.dot(voted[k].first) > 0 ? 1 : -1;
            if (y_hat == y_i) {
                voted[k].second++;
            } else {
                Eigen::Vector3d v_kplus1(0, 0, 0);
                v_kplus1 = voted[k].first + y_i * x_i;
                int c_kplus1 = 1;
                voted.push_back({v_kplus1, c_kplus1});
                k++;
            }
        }
    }
    cout << "Voted Form" << endl;
    return voted;
}
void predict_voted(vector<pair<Eigen::Vector3d, int>> voted,
                   Eigen::Vector2d x) {
    Eigen::Vector3d x_homo(0, 0, 0);
    x_homo.head(2) = x;
    x_homo(2) = 1;
    double sum = 0;
    for (int i = 0; i < voted.size(); i++) {
        double _sign = voted[i].first.dot(x_homo) > 0 ? 1 : -1;
        sum += voted[i].second * _sign;
    }
    double y_hat = sum > 0 ? 1 : -1;
    cout << "predicted: y_hat = " << y_hat << endl;
}

int main(int argc, char const* argv[]) {
    perceptron model = perceptron();
    int cols = 500;
    Eigen::Matrix2Xd x_train(2, cols);
    Eigen::VectorXd y_train(cols);
    model.readData(x_train, y_train);
    // cout<<x_train<<endl;
    // model.readIris(x_train, y_train);
    Eigen::MatrixXd gram(cols, cols);

    clock_t time_stt = clock();
    // origin form
    model.fit(x_train, y_train);
    cout << "time of 'Origin Form of Perceptron' is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;

    Eigen::Vector2d w = model.getWeight();
    double bias = model.getBias();
    cout << "weight: " << w.transpose() << endl;
    cout << "bias: " << bias << endl;
    model.clear();

    // duality form
    model.getGramMatrix(x_train, gram);
    cout << "gram computed" << endl;
    time_stt = clock();
    model.fit_duality(gram, x_train, y_train);
    cout << "time of 'Duality Form of Perceptron' is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;

    Eigen::Vector2d w2 = model.getWeight();
    double bias2 = model.getBias();
    cout << "weight: " << w2.transpose() << endl;
    cout << "bias: " << bias2 << endl;

    cout << endl;
    int T = 100;
    cout << "epoches T=" << T << endl;
    // voted form
    time_stt = clock();
    vector<pair<Eigen::Vector3d, int>> voted =
        model.fit_voted(T, x_train, y_train);
    cout << "time of 'Voted Form of Perceptron' is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;
    for (int i = 0; i < voted.size(); i++) {
        cout << voted[i].first.transpose() <<"; c=" <<voted[i].second<<endl;
    }
    Eigen::Vector2d test(-8.641,6.976);
    predict_voted(voted, test);
}
