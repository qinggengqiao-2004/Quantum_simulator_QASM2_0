import json
import re
import math
import numpy as np # 需要引入 numpy 辅助处理矩阵

class QASMCompiler:
    def __init__(self, noise_model=None):
        """
        初始化编译器。
        :param noise_model: 噪声配置字典，例如 {'CX': 0.01, 'U': 0.001}
        """
        self.gate_defs = {}
        self.noise_model = noise_model or {}
        
        # 注册 OpenQASM 2.0 的底层不透明原语 (Opaque primitives)
        self.gate_defs['U'] = {'params': ['theta', 'phi', 'lambda'], 'qubits': ['q'], 'body': 'PRIMITIVE'}
        self.gate_defs['CX'] = {'params': [], 'qubits': ['c', 't'], 'body': 'PRIMITIVE'}

    def _strip_comments(self, qasm_str):
        """移除单行注释并清理空白"""
        return re.sub(r'//.*', '', qasm_str)

    def load_inc(self, content):
        """解析 qelib1.inc 文本内容"""
        clean_content = self._strip_comments(content)
        
        # 匹配 gate name(params) q1, q2 { body }
        gate_pattern = re.compile(r'gate\s+([a-zA-Z0-9_]+)(?:\((.*?)\))?\s+([a-zA-Z0-9_\s,]+)\s*\{(.*?)\}', re.DOTALL)
        
        for name, params_str, qubits_str, body_str in gate_pattern.findall(clean_content):
            params = [p.strip() for p in params_str.split(',')] if params_str.strip() else []
            qubits = [q.strip() for q in qubits_str.split(',')]
            
            body = []
            # 以分号分割指令
            for line in body_str.strip().split(';'):
                line = line.strip()
                if not line: continue
                # 匹配指令调用，例如 u1((lambda+phi)/2) c
                inst_match = re.match(r'([a-zA-Z0-9_]+)(?:\((.*?)\))?\s+(.*)', line)
                if inst_match:
                    g_name, g_params_str, g_qubits_str = inst_match.groups()
                    g_params = [p.strip() for p in g_params_str.split(',')] if g_params_str else []
                    g_qubits = [q.strip() for q in g_qubits_str.split(',')]
                    body.append((g_name, g_params, g_qubits))
            
            self.gate_defs[name] = {'params': params, 'qubits': qubits, 'body': body}

    def _eval_math(self, expr, param_map):
        """安全地计算 QASM 参数中的数学表达式"""
        if not isinstance(expr, str):
            return float(expr)
        
        # 替换 pi 为具体数值
        expr = expr.replace('pi', str(math.pi))
        
        # 将形参替换为具体数值 (使用正则确保只替换完整单词)
        for k, v in param_map.items():
            expr = re.sub(rf'\b{k}\b', str(v), expr)
            
        try:
            # 基础运算安全求值
            return eval(expr, {"__builtins__": None}, {})
        except Exception as e:
            return expr # 如果解析失败返回原字符串，方便调试

    def decompose_gate(self, name, params, qubits):
        """递归分解量子门，并注入噪声通道"""
        if name not in self.gate_defs:
            raise ValueError(f"Unknown gate: {name}")
        
        definition = self.gate_defs[name]
        
        # --- 到底层原语，生成目标指令和噪声 ---
        if definition['body'] == 'PRIMITIVE':
            # 为了 C++ 方便，统一转化为小写的 u3 和 cx 标识
            mapped_name = 'u3' if name == 'U' else 'cx'
            ops = [{'gate': mapped_name, 'params': params, 'qubits': qubits}]
            
            # 注入去极化噪声模型 (Depolarizing Noise)
            if name in self.noise_model:
                prob = self.noise_model[name]
                if prob > 0:
                    ops.append({
                        'gate': 'depolarizing_error',
                        'prob': prob,
                        'qubits': qubits
                    })
            return ops
        
        # --- 递归展开非底层门 ---
        qubit_map = dict(zip(definition['qubits'], qubits))
        param_map = dict(zip(definition['params'], params))
        
        results = []
        for sub_name, sub_params_expr, sub_qubits in definition['body']:
            mapped_qubits = [qubit_map[q] for q in sub_qubits]
            # 计算这一层的参数
            mapped_params = [self._eval_math(p, param_map) for p in sub_params_expr]
            # 继续递归
            results.extend(self.decompose_gate(sub_name, mapped_params, mapped_qubits))
        
        return results

    import numpy as np # 需要引入 numpy 辅助处理矩阵

# ... [保留原本的 QASMCompiler 的 __init__, load_inc, _eval_math, decompose_gate 等方法] ...

    def compile(self, qasm_code, output_file, initial_rho=None):
        """
        编译代码并输出 JSON，支持显式输出初始密度矩阵
        :param qasm_code: QASM 字符串
        :param output_file: 输出的 JSON 文件路径
        :param initial_rho: (可选) numpy array 格式的初始密度矩阵
        """
        compiled_ops = []
        clean_code = self._strip_comments(qasm_code)
        
        num_qubits = 0
        
        for line in clean_code.split(';'):
            line = line.strip()
            if not line or line.startswith('include'): continue
            
            # 解析 qreg 提前获取系统比特数 N
            if line.startswith('qreg'):
                match = re.match(r'qreg\s+[a-zA-Z0-9_]+\[(\d+)\]', line)
                if match:
                    num_qubits = max(num_qubits, int(match.group(1)))
                continue
            
            match = re.match(r'([a-zA-Z0-9_]+)(?:\((.*?)\))?\s+(.*)', line)
            if match:
                name, p_str, q_str = match.groups()
                params = [self._eval_math(p.strip(), {}) for p in p_str.split(',')] if p_str else []
                qubits = [q.strip() for q in q_str.split(',')]
                compiled_ops.extend(self.decompose_gate(name, params, qubits))
        
        # 容错机制：如果没有 qreg 定义，则扫描所有的门操作推断最大比特索引
        if num_qubits == 0 and compiled_ops:
            max_index = -1
            for op in compiled_ops:
                for q_str in op['qubits']:
                    m = re.search(r'\[(\d+)\]', q_str)
                    if m:
                        max_index = max(max_index, int(m.group(1)))
            num_qubits = max_index + 1
            
        # 计算系统维度 dim = 2^N
        dim = 2 ** num_qubits

        # 处理初始密度矩阵
        if initial_rho is not None:
            # 用户提供了自定义的 numpy 矩阵，转为标准列表格式
            formatted_rho = [[[float(initial_rho[i, j].real), float(initial_rho[i, j].imag)] 
                              for j in range(initial_rho.shape[1])] 
                             for i in range(initial_rho.shape[0])]
        else:
            # 未提供初始态时，显式生成 |0...0><0...0| 纯态
            formatted_rho = [[[0.0, 0.0] for _ in range(dim)] for _ in range(dim)]
            if dim > 0:
                formatted_rho[0][0] = [1.0, 0.0]  # 左上角对角线元素设为 1

        # 构建最终输出的字典
        output_data = {
            "initial_state": formatted_rho,
            "operations": compiled_ops
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Compilation finished. JSON exported to {output_file}")


# ==========================================
# 使用演示
# ==========================================

# 1. 定义硬件的噪声模型 (设置错误率 p)
hardware_noise = {
    'U': 0.001,  # 单比特门错误率 0.1%
    'CX': 0.01   # 双比特门错误率 1%
}

compiler = QASMCompiler(noise_model=hardware_noise)

# 2. 读取你提供的 qelib1.inc 字符串 (此处假设已经保存在变量 qelib_content 中)
with open('lib/qelib1.inc', 'r') as f:
    qelib_content = f.read()
compiler.load_inc(qelib_content)

# 3. 编译一段含复杂门的测试代码：Toffoli门和受控U门

'''
test_qasm = """
include "qelib1.inc";
qreg q[3];
ccx q[0], q[1], q[2];
cu3(pi, pi/2, pi/4) q[0], q[1];
"""
'''

with open('QFT_3qbit.qasm', 'r') as f:
    test_qasm =  f.read()

compiler.compile(test_qasm, "noisy_simulation_circuit.json")