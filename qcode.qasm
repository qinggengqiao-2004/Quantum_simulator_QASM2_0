//OPENQASM 2.0;
include "qelib1.inc";

// 1. 声明寄存器
qreg q[3];
//creg c[3];

// 2. 核心 QFT 操作：从最高有效位（q[0]）开始
// 对第 0 个量子比特施加 Hadamard 门和受控相位旋转
//reset q[0];
//reset q[1];
//reset q[2];
h q[0];
cu1(pi / 2) q[1], q[0];
cu1(pi / 4) q[2], q[0];

// 对第 1 个量子比特施加 Hadamard 门和受控相位旋转
h q[1];
cu1(pi / 2) q[2], q[1];

// 对第 2 个量子比特施加 Hadamard 门
h q[2];

// 3. 交换操作 (SWAP) 以匹配标准端序输出
// 使用三次 CNOT 门实现 q[0] 和 q[2] 的 SWAP
cx q[0], q[2];
cx q[2], q[0];
cx q[0], q[2];

// 4. 测量操作
//measure q[0] -> c[0];
//measure q[1] -> c[1];
//measure q[2] -> c[2];
