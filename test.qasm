
//OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
// 这是一个制备贝尔态的线路
h q[0];
rx(pi/2) q[1];
cx q[0], q[1];

