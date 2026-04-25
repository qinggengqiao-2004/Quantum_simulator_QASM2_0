#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <regex>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

using json = nlohmann::json;
using namespace Eigen;
using namespace std::complex_literals;

// 提取 "q[x]" 中的整数 x
int parse_qubit(const std::string& q_str) {
    std::regex re(R"(q\[(\d+)\])");
    std::smatch match;
    if (std::regex_search(q_str, match, re)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

// 将整数状态转为二进制字符串格式（q[0] 为最高有效位 MSB）
std::string to_binary_string(int val, int N) {
    std::string s;
    for (int i = N - 1; i >= 0; --i) {
        s += ((val >> i) & 1) ? "1" : "0";
    }
    return s;
}

// 将单比特矩阵扩展到 N 比特系统的 2^N x 2^N 矩阵
MatrixXcd expand_1q_matrix(const Matrix2cd& U, int target, int N) {
    MatrixXcd res = MatrixXcd::Identity(1, 1);
    Matrix2cd I2 = Matrix2cd::Identity();
    // 采用 Big-Endian 约定：q[0] 是张量乘积的最左侧（MSB）
    for (int i = 0; i < N; ++i) {
        if (i == target) {
            res = kroneckerProduct(res, U).eval();
        } else {
            res = kroneckerProduct(res, I2).eval();
        }
    }
    return res;
}

// 将两个单比特矩阵作用在双比特上，并扩展到 N 比特系统
MatrixXcd expand_2q_matrix(const Matrix2cd& U1, int q1, const Matrix2cd& U2, int q2, int N) {
    MatrixXcd res = MatrixXcd::Identity(1, 1);
    Matrix2cd I2 = Matrix2cd::Identity();
    for (int i = 0; i < N; ++i) {
        if (i == q1) res = kroneckerProduct(res, U1).eval();
        else if (i == q2) res = kroneckerProduct(res, U2).eval();
        else res = kroneckerProduct(res, I2).eval();
    }
    return res;
}

// 构建受控非门 CX 的 2^N x 2^N 全局矩阵
MatrixXcd build_cx(int control, int target, int N) {
    int dim = 1 << N;
    MatrixXcd CX = MatrixXcd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        // 读取 control 比特的值
        bool ctrl_val = (i >> (N - 1 - control)) & 1;
        int j = i;
        if (ctrl_val) {
            j = i ^ (1 << (N - 1 - target)); // 翻转 target 比特
        }
        CX(j, i) = 1.0;
    }
    return CX;
}

// 构建 U3 门的 2x2 矩阵
Matrix2cd build_u3(double theta, double phi, double lambda) {
    Matrix2cd U;
    std::complex<double> i_comp(0, 1);
    U(0, 0) = std::cos(theta / 2.0);
    U(0, 1) = -std::exp(i_comp * lambda) * std::sin(theta / 2.0);
    U(1, 0) = std::exp(i_comp * phi) * std::sin(theta / 2.0);
    U(1, 1) = std::exp(i_comp * (phi + lambda)) * std::cos(theta / 2.0);
    return U;
}


// [保留原来的 parse_qubit, to_binary_string, expand_1q_matrix, expand_2q_matrix, build_cx, build_u3 函数...]

int main(int argc, char* argv[]) {
    std::string filename = "noisy_simulation_circuit.json";
    if (argc > 1) filename = argv[1];

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open JSON file: " << filename << std::endl;
        return 1;
    }

    json doc;
    file >> doc;

    // 为了兼容旧版本纯数组的情况，判断一下最外层是不是数组
    json operations = doc.is_array() ? doc : doc["operations"];

    // 1. 获取量子比特数量 N
    int max_q = 0;
    for (const auto& op : operations) {
        for (const auto& q_str : op["qubits"]) {
            max_q = std::max(max_q, parse_qubit(q_str.get<std::string>()));
        }
    }
    int N = max_q + 1;
    int dim = 1 << N;
    std::cout << "Simulating " << N << " qubits. Matrix dimension: " << dim << "x" << dim << std::endl;

    // 2. 初始化密度矩阵 rho
    MatrixXcd rho = MatrixXcd::Zero(dim, dim);

    // 判断并读取 JSON 中的初始密度矩阵
    if (!doc.is_array() && doc.contains("initial_state") && !doc["initial_state"].is_null()) {
        auto init_state = doc["initial_state"];
        
        // 校验输入矩阵维度是否合法
        int input_dim = init_state.size();
        if (input_dim != dim) {
            std::cerr << "Warning: The dimension of initial_state (" << input_dim << "x" << input_dim 
                      << ") does not match the required dimension (" << dim << "x" << dim 
                      << ") based on operations. Falling back to |0><0| state." << std::endl;
            rho(0, 0) = 1.0;
        } else {
            std::cout << "Loading custom initial density matrix from JSON..." << std::endl;
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    double real_part = init_state[i][j][0].get<double>();
                    double imag_part = init_state[i][j][1].get<double>();
                    rho(i, j) = std::complex<double>(real_part, imag_part);
                }
            }
        }
    } else {
        std::cout << "No custom initial state found. Using default |0...0><0...0| state." << std::endl;
        rho(0, 0) = 1.0; // 默认的 |0...0><0...0| 态
    }

    // 准备泡利矩阵（用于去极化噪声）
    Matrix2cd Pauli[4];
    Pauli[0] << 1, 0, 0, 1;   // I
    Pauli[1] << 0, 1, 1, 0;   // X
    Pauli[2] << 0, -1i, 1i, 0; // Y
    Pauli[3] << 1, 0, 0, -1;  // Z

    // 3. 逐条执行门指令
    for (const auto& op : operations) {
        std::string gate = op["gate"];
        
        if (gate == "u3") {
            double theta = op["params"][0];
            double phi = op["params"][1];
            double lambda = op["params"][2];
            int q = parse_qubit(op["qubits"][0]);
            
            Matrix2cd U = build_u3(theta, phi, lambda);
            MatrixXcd U_full = expand_1q_matrix(U, q, N);
            
            rho = U_full * rho * U_full.adjoint();
        } 
        else if (gate == "cx") {
            int c = parse_qubit(op["qubits"][0]);
            int t = parse_qubit(op["qubits"][1]);
            
            MatrixXcd CX_full = build_cx(c, t, N);
            rho = CX_full * rho * CX_full.adjoint();
        }
        else if (gate == "depolarizing_error") {
            double p = op["prob"];
            auto qubits = op["qubits"];
            
            if (qubits.size() == 1) {
                int q = parse_qubit(qubits[0]);
                MatrixXcd rho_new = (1.0 - p) * rho;
                for (int k = 1; k < 4; ++k) {
                    MatrixXcd P_full = expand_1q_matrix(Pauli[k], q, N);
                    rho_new += (p / 3.0) * (P_full * rho * P_full.adjoint());
                }
                rho = rho_new;
            } 
            else if (qubits.size() == 2) {
                int q1 = parse_qubit(qubits[0]);
                int q2 = parse_qubit(qubits[1]);
                MatrixXcd rho_new = (1.0 - p) * rho;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        if (i == 0 && j == 0) continue; 
                        MatrixXcd P_full = expand_2q_matrix(Pauli[i], q1, Pauli[j], q2, N);
                        rho_new += (p / 15.0) * (P_full * rho * P_full.adjoint());
                    }
                }
                rho = rho_new;
            }
        }
    }

    // 4. 将密度矩阵保存到 CSV
    std::ofstream dm_file("density_matrix.csv");
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            dm_file << rho(i, j).real() << (rho(i, j).imag() >= 0 ? "+" : "") << rho(i, j).imag() << "i";
            if (j < dim - 1) dm_file << ",";
        }
        dm_file << "\n";
    }
    dm_file.close();
    std::cout << "Density matrix saved to 'density_matrix.csv'." << std::endl;

    // 5. 将每个态的测量概率保存到 CSV (概率 P(k) 即密度矩阵的对角线元素)
    std::ofstream prob_file("probabilities.csv");
    prob_file << "State,Binary,Probability\n";
    for (int i = 0; i < dim; ++i) {
        double prob = std::abs(rho(i, i));
        prob_file << i << "," << to_binary_string(i, N) << "," << std::fixed << std::setprecision(6) << prob << "\n";
    }
    prob_file.close();
    std::cout << "Measurement probabilities saved to 'probabilities.csv'." << std::endl;

    std::cout << "Simulation completed." << std::endl;
    return 0;
}
