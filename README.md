Optimal Multi-Period Production Planning for Cost Minimization

Team Name: Optimal Trio
Members

BT2024231 - Advaya Bhardwaj

BT2024014 - Sindhoor Ganapathi Hegde

BT2024123 -10	 Aman Kumar Lahoti

Project Title

Optimal Multi-Period Production Planning for Cost Minimization

**Short Description**

This project addresses a real-world production planning problem using Linear Programming (LP).
A manufacturing company must determine optimal monthly production levels for multiple products while minimizing total cost over a six-month time period.

The model incorporates:
Monthly product demand
Limited production capacity
Production cost
Inventory holding cost
Backlog penalty cost
Setup cost per month
The optimization model outputs:
Production quantity per month
Inventory carried over
Backlog (unmet demand)
Setup decisions

The model is implemented in Python using the PuLP optimizer and includes scenario analysis to study the impact of demand, capacity, and cost variations.

**Dataset Overview**

The project uses three CSV files, stored in the Datasets/ directory:
1. 01_products.csv
Contains product-level information:
Product name
Production cost (CProd)
Inventory cost (CInv)
Backlog cost (CBack)
Resource requirement per unit (RRes)
Selling price

2. 02_demand.csv
Contains monthly demand for each product (July to December).

3. 03_capacity.csv

Contains the total available production hours for each month.

**Model Choice: Linear Programming (LP)**
Linear Programming was chosen because:
Decision variables (production, inventory, backlog) are continuous.
Setup variable is binary, enabling Mixed Integer Linear Programming.
Objective function is linear.
Constraints (capacity, inventory balance, backlog limits) are linear.
LP solvers such as CBC (used via PuLP) are efficient and reliable.

**Objective Function**
Minimize
∑(𝐶𝑃𝑟𝑜𝑑⋅𝑃𝑝,𝑚)+∑(𝐶𝐼𝑛𝑣⋅𝐼𝑝,𝑚)+∑(𝐶𝐵𝑎𝑐𝑘⋅𝐵𝑝,𝑚)+∑(𝐶𝑆𝑒𝑡𝑢𝑝⋅𝑌𝑚)Minimize ∑(CProd⋅Pp,m)+∑(CInv⋅Ip,m)+∑(CBack⋅Bp,m)+∑(CSetup⋅Ym)
Where:

Pp,m: Production for product p in month 𝑚
Ip,m: Inventory at end of month
Bp,m:Backlog
𝑌𝑚 :Setup decision (1 if production happens in month 𝑚)

**Constraints**
1.Inventory balance
 	Im−1+Pm−Dm=Im​−Bm
 	​

2. Capacity Constraint
 	∑(RResp​⋅Pp,m​)≤Capacitym​

3. Backlog Limit
 	Bp,m​≤Demandp,m​

4.Setup Constraint (Big-M)
 	Pp,m​≤Mp,m​⋅Ym​


**Libraries Used**
pandas — Data loading and preprocessing
pulp — Optimization solver (Linear Programming)
matplotlib, seaborn — Data visualization
math, os, copy — Utility functions


**Setup Instructions**
1.Install required libraries:
pip install pandas pulp matplotlib seaborn

2. Dataset structure
Ensure the following folder structure exists:
Datasets/
 ├── 01_products.csv
 ├── 02_demand.csv
 └── 03_capacity.csv
3.Run the Optimization Model
  python production_planning_model.py
4. Run the visualization 
  python version2.py



 	​

 	​
