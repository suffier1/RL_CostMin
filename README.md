# RL_CostMin
This work is for 대한전자공학회 2025년도 하계종합학술대회 인공지능 학부생 논문 경진대회.<br/>
link: https://bpai2025.github.io/ <br/>
Paper Title: Gate-Level Cell Selection via Reinforcement Learning for Netlist Cost Minimization <br/>

## netlist_processor.py
For pharsing verilog, evaluate cost by cost_estimator.. etc (cost_estimators are in release folder)

## main.py
A3C Algorithm, Segmenting.. etc

## release
Provided by ICCAD Contest 2024 Prob.A (https://www.iccad-contest.org/2024/Problems)

## How to simulate
Store netlist_processor.py, main.py, release in same directory
#### Example
!python main.py \
  -d 4 \
  -c 5 \
  --seg_size 100 \
  --first_ep 3500 \
  --other_ep 10 \
  --workers 16  <br/>
  <br/>
d : design number (inside release folder) <br/>
c : cost estimator number (also inside relase folder) <br/>
seg_size: number of gates inside one size. If #of all gates are 1000 and seg_size = 100, # of segments = 10 <br/>
first_ep: number of episodes for first segment <br/>
other_ep: number of episodes for every segment except first <br/>
workers: number of workers in A3C Algorithm <br/>

Note: The training was conducted in a Colab Pro+ environment.
