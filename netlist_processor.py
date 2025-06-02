import re, json, random, subprocess, tempfile
from dataclasses import dataclass
from typing import List
import logging
from collections import defaultdict

GATE_TYPE_MAP = {"and": 0, "buf": 1, "nand": 2, "nor": 3, "not": 4, "or": 5, "xnor": 6, "xor": 7}

@dataclass
class Gate:
    base_type: str         # 예: "or", "and", 등
    sub_num: int           # netlist에서 추출된 숫자
    name: str              # 예: "g0"
    inputs: List[str]
    output: str
    order: int

@dataclass
class Netlist:
    module_name: str
    inputs: List[str]
    outputs: List[str]
    wires: List[str]
    gates: List[Gate]

# 외부에서 set 해주는 전역 LIBRARY_CELLS
LIBRARY_CELLS = []

def load_library(lib_filepath: str) -> List[dict]:
    with open(lib_filepath, 'r') as file:
        library = json.load(file)
    return library.get("cells", [])

def lib_gate_nums(library_path):
    """
    라이브러리에서 게이트별 sub_num 후보들을 모아둔다.
    예: {
      "and": [1,2,3,...],
      "or": [1,2,3,...],
      ...
    }
    """
    with open(library_path, "r") as f:
        data = json.load(f)
        
        gate_numbers = {}
        for cell in data["cells"]:
            cell_name = cell.get("cell_name","")
            parts = cell_name.split("_")
            if len(parts) <2:
                continue

            gate_type = parts[0]
            num_part = parts[1]
            num = int(num_part)

            if gate_type not in gate_numbers:
                gate_numbers[gate_type] = []
            gate_numbers[gate_type].append(num)

        for key in gate_numbers:
            gate_numbers[key].sort()
    return gate_numbers

def num_matching_random(cell_type: str, library: List[dict]) -> int:
    """
    주어진 cell_type(예: "and")에 대해 라이브러리에 들어 있는 "and_1", "and_2"... 중 무작위 선택.
    """
    ctype = cell_type.strip().lower()
    matching = [
        cell["cell_name"]
        for cell in library
        if cell.get("cell_type", "").strip().lower() == ctype
    ]
    if not matching:
        raise ValueError(f"No matching cells found for type: {cell_type}")
    chosen = random.choice(matching)  # 예: "and_3"
    try:
        return int(chosen.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid cell_name format for cell: {chosen}")

def parse_gate_line(line: str, order: int) -> Gate:
    """
    한 게이트 라인 예:
      or_2   g1 (out1, in1, in2)
    """
    global LIBRARY_CELLS
    line = line.strip()
    pattern = r'^\s*([a-zA-Z]+)_?(\d*)\s+(\w+)\s*\(\s*([^)]*?)\s*\)\s*$'
    m = re.match(pattern, line)
    if not m:
        raise ValueError(f"Cannot parse gate line: {line}")
    
    base_type_raw, sub_num_str, inst_name, contents = m.groups()
    base_type = base_type_raw.strip().lower()
    
    if sub_num_str:
        sub_num = int(sub_num_str)
    else:
        if not LIBRARY_CELLS:
            raise ValueError("LIBRARY_CELLS is empty. Please load the cell library before parsing the netlist.")
        sub_num = num_matching_random(base_type, LIBRARY_CELLS)
    
    tokens = [token.strip() for token in contents.split(',')]
    if len(tokens) < 2:
        raise ValueError(f"Insufficient pins in line: {line}")
    output = tokens[0]
    inputs = tokens[1:]
    
    return Gate(
        base_type=base_type,
        sub_num=sub_num,
        name=inst_name,
        inputs=inputs,
        output=output,
        order=order
    )

def parse(filepath: str) -> Netlist:
    """
    Netlist 파일(.v)을 파싱하여 Netlist 객체를 반환.
    """
    gates = []
    module_name = ""
    inputs = []
    outputs = []
    wires = []
    
    with open(filepath, 'r') as f:
        netlist_text = f.read()
    
    items = netlist_text.split(';')
    gate_order = 0
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.startswith("module"):
            module_name = re.sub(r'(\(|//|\n).*', '', item).replace("module", "").strip()
        elif item.startswith("input"):
            inputs.extend([s.strip() for s in item.replace("input", "").strip().split(',') if s.strip()])
        elif item.startswith("output"):
            outputs.extend([s.strip() for s in item.replace("output", "").strip().split(',') if s.strip()])
        elif item.startswith("wire"):
            wires.extend([s.strip() for s in item.replace("wire", "").strip().split(',') if s.strip()])
        elif item.startswith("endmodule"):
            continue
        else:
            gate = parse_gate_line(item, gate_order)
            gates.append(gate)
            gate_order += 1
    return Netlist(module_name=module_name, inputs=inputs, outputs=outputs, wires=wires, gates=gates)

def generate_verilog_forCE(netlist: Netlist) -> str:
    """
    cost estimator에 전달할 Verilog 생성.
    게이트의 인스턴스 이름은 {base_type}_{sub_num}로 사용한다.
    """
    lines = []
    all_ports = netlist.inputs + netlist.outputs

    def sort_key(s):
        m = re.match(r'n(\d+)', s)
        return int(m.group(1)) if m else 999999
    sorted_ports = sorted(all_ports, key=sort_key)
    
    lines.append(f"module {netlist.module_name} ({', '.join(sorted_ports)});")
    lines.append(f"  input {', '.join(netlist.inputs)};")
    lines.append(f"  output {', '.join(netlist.outputs)};")
    
    for g in sorted(netlist.gates, key=lambda x: x.order):
        # Verilog에서 게이트 인스턴스는 "base_sub inst(...);" 형태
        pin_list = g.inputs + [g.output]
        lines.append(f"  {g.base_type}_{g.sub_num} {g.name}({','.join(pin_list)});")
    
    lines.append("endmodule")
    return "\n".join(lines)

def cost(netlist: Netlist, cost_estimator_path: str, library_path: str) -> float:
    """
    임시 netlist 파일 -> cost_estimator 실행 -> stdout 또는 output 파일에서 "cost = 값" 찾기
    """
    verilog_str = generate_verilog_forCE(netlist)

    # 1) 임시 netlist 파일 생성
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".v", delete=False) as tmp_netlist:
        tmp_netlist.write(verilog_str)
        tmp_netlist_path = tmp_netlist.name

    # 2) 임시 output 파일
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    cmd = [
        cost_estimator_path,
        "-library", library_path,
        "-netlist", tmp_netlist_path,
        "-output", tmp_out_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 우선 stdout에서 "cost =" 부분 찾기
    output = result.stdout.strip()
    cost_value = None

    lines = output.split('\n')
    for line in lines:
        if "cost =" in line:
            # e.g. "cost = 1.23"
            cost_str = line.split('=')[1].strip()
            try:
                cost_value = float(cost_str)
                break
            except ValueError:
                logging.warning(f"Found 'cost =' but parsing float failed in line: {line}")

    # stdout에서 못 찾으면, output 파일 읽어서 다시 찾기
    if cost_value is None:
        try:
            with open(tmp_out_path, "r") as f:
                file_output = f.read().strip()
                for line in file_output.split('\n'):
                    if "cost =" in line:
                        cost_str = line.split('=')[1].strip()
                        cost_value = float(cost_str)
                        break
        except Exception as e:
            logging.error(f"[cost] Failed to read cost from file {tmp_out_path}: {e}")
    
    # 여전히 cost_value가 None이면 예외
    if cost_value is None:
        raise ValueError(f"[cost] Could not parse 'cost =' from either stdout or {tmp_out_path}.\n"
                         f"stdout:\n{output}\n")
    
    return cost_value

# def make_graph(netlist: Netlist):
#     """
#     (추가) netlist로부터 그래프(인접 리스트), gate_index_map을 만드는 함수
#     """
#     graph = {}
#     gate_index_map = {}
#     signal_to_gate_ids = {}

#     # 노드 초기화 및 입력 신호 매핑
#     for gate in netlist.gates:
#         node_id = gate.order
#         graph[node_id] = []
#         gate_index_map[gate.name] = node_id
#         for inp in gate.inputs:
#             if inp not in signal_to_gate_ids:
#                 signal_to_gate_ids[inp] = []
#             signal_to_gate_ids[inp].append(node_id)

#     # 각 게이트의 출력 신호 -> 해당 신호를 입력으로 갖는 게이트들
#     for gate in netlist.gates:
#         src_id = gate.order
#         dst_gate_ids = signal_to_gate_ids.get(gate.output, [])
#         for dst_id in dst_gate_ids:
#             if src_id != dst_id:
#                 graph[src_id].append(dst_id)
#     return graph, gate_index_map

def make_graph_node(netlist: Netlist, library_map: dict):
    """
     make_graph(dataclass 형태의 Netlist, library_datas(함수 사용해서 만들면 됨)) 넣어주면 graph 생성
     """
    
    node = {}

    for gate in netlist.gates:
        cell_name = f"{gate.base_type}_{gate.sub_num}"
        features = library_map.get(cell_name)
        node[gate.order] = [GATE_TYPE_MAP[gate.base_type], gate.sub_num] + features

    return node


def make_graph_edge(netlist):
    input_signal_to_gate_ids = {}
    for gate in netlist.gates:
        for inp in gate.inputs:
            input_signal_to_gate_ids.setdefault(inp, []).append(gate.order)
    
    edge= {}
    for gate in netlist.gates:
        edge[gate.order] = input_signal_to_gate_ids.get(gate.output, [])
    return edge







def library_datas(library_path):
    """
    각 cell_name마다의 특징(data1~7)을 lib 형태로 return한다.
    """
    with open(library_path,'r') as file:
        library = json.load(file)
        
    lib_data_map = {}

    for cell in library.get("cells"):
        cell_name = cell.get("cell_name")
        lib_data_map[cell_name] = [
            float(cell["data_1_f"]),
                float(cell["data_2_f"]),
                float(cell["data_3_i"]),
                float(cell["data_4_f"]),
                float(cell["data_5_f"]),
                float(cell["data_6_f"]),
                float(cell["data_7_f"])
        ]

    return lib_data_map