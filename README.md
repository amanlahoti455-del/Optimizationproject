# 🏭 Optimal Multi-Period Production Planning for Cost Minimization

> **Team:** Optimal Trio  
> **Members:**  
> - BT2024231 — Advaya Bhardwaj  
> - BT2024014 — Sindhoor Ganapathi Hegde  
> - BT2024123 — Aman Kumar Lahoti  

---

## 📌 Overview

This project solves a real-world **multi-period production planning problem** using **Mixed Integer Linear Programming (MILP)**. A manufacturing company must determine the optimal monthly production levels for multiple products over a six-month horizon (July–December), minimizing total operational costs.

The model jointly optimizes:

- Monthly production quantities per product
- Inventory carried over between months
- Backlog (unmet demand) allowed under penalty
- Binary setup decisions (whether production runs in a given month)

---

## 🎯 Objective Function

Minimize total cost across all products `p` and months `m`:

$$\text{Minimize} \sum_{p,m}(C_{\text{Prod}} \cdot P_{p,m}) + \sum_{p,m}(C_{\text{Inv}} \cdot I_{p,m}) + \sum_{p,m}(C_{\text{Back}} \cdot B_{p,m}) + \sum_{m}(C_{\text{Setup}} \cdot Y_m)$$

| Symbol | Description |
|--------|-------------|
| $P_{p,m}$ | Production quantity for product $p$ in month $m$ |
| $I_{p,m}$ | Inventory held at end of month $m$ for product $p$ |
| $B_{p,m}$ | Backlog (unmet demand) for product $p$ in month $m$ |
| $Y_m$ | Binary setup decision — 1 if any production occurs in month $m$ |
| $C_{\text{Prod}}$ | Per-unit production cost |
| $C_{\text{Inv}}$ | Per-unit inventory holding cost |
| $C_{\text{Back}}$ | Per-unit backlog penalty cost |
| $C_{\text{Setup}}$ | Fixed cost incurred when production is set up in a month |

---

## 📐 Constraints

**1. Inventory Balance** — ensures flow conservation each month:

$$I_{p,m-1} + P_{p,m} - D_{p,m} = I_{p,m} - B_{p,m}$$

**2. Capacity Constraint** — total resource usage must not exceed available capacity:

$$\sum_{p}(R_{\text{Res},p} \cdot P_{p,m}) \leq \text{Capacity}_m$$

**3. Backlog Limit** — backlog cannot exceed actual demand:

$$B_{p,m} \leq D_{p,m}$$

**4. Setup Constraint (Big-M)** — production in a month requires a setup:

$$P_{p,m} \leq M_{p,m} \cdot Y_m$$

---

## 📂 Project Structure

```
project/
│
├── Datasets/
│   ├── 01_products.csv       # Product-level cost and resource data
│   ├── 02_demand.csv         # Monthly demand per product (Jul–Dec)
│   └── 03_capacity.csv       # Available production hours per month
│
├── production_planning_model.py   # Core MILP optimization model
├── version2.py                    # Visualization and scenario analysis
└── README.md
```

---

## 🗃️ Dataset Description

### `01_products.csv`
| Column | Description |
|--------|-------------|
| `Product` | Product name/ID |
| `CProd` | Production cost per unit |
| `CInv` | Inventory holding cost per unit per month |
| `CBack` | Backlog penalty cost per unit per month |
| `RRes` | Resource hours required per unit |
| `SellingPrice` | Selling price per unit |

### `02_demand.csv`
Monthly demand for each product across the 6-month planning horizon (July–December).

### `03_capacity.csv`
Total available production hours per month across the planning horizon.

---

## ⚙️ Why Linear Programming?

| Criterion | Justification |
|-----------|---------------|
| Decision variables | Production, inventory, and backlog are continuous — LP-compatible |
| Setup variable | Binary (0/1) — extends LP to **Mixed Integer LP (MILP)** |
| Objective | Linear combination of cost components |
| Constraints | All constraints are linear (balance, capacity, backlog, Big-M) |
| Solver | CBC via PuLP — reliable, open-source, and efficient for this scale |

---

## 🛠️ Setup & Installation

**1. Install dependencies:**

```bash
pip install pandas pulp matplotlib seaborn
```

**2. Verify dataset structure:**

```
Datasets/
├── 01_products.csv
├── 02_demand.csv
└── 03_capacity.csv
```

**3. Run the optimization model:**

```bash
python production_planning_model.py
```

**4. Run visualizations & scenario analysis:**

```bash
python version2.py
```

---

## 📊 Model Outputs

After solving, the model reports:

- **Production schedule** — units to produce per product per month
- **Inventory levels** — stock carried over at end of each month
- **Backlog** — unmet demand incurred and its cost
- **Setup decisions** — which months require production setup
- **Total minimized cost** — breakdown across all cost components
- **Scenario analysis** — impact of varying demand, capacity, and cost parameters

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and preprocessing |
| `pulp` | MILP optimization (CBC solver) |
| `matplotlib` | Plotting production and cost charts |
| `seaborn` | Enhanced statistical visualizations |
| `math`, `os`, `copy` | Utility functions |

---

## 🔍 Scenario Analysis

The model includes scenario testing to evaluate sensitivity to:

- **Demand changes** — higher/lower demand across products
- **Capacity fluctuations** — constrained vs. relaxed production hours
- **Cost variations** — impact of changing holding, backlog, or setup costs

This helps decision-makers understand tradeoffs and build robust production plans.

---

## 📝 Notes

- The planning horizon covers **6 months: July through December**.
- Initial inventory and backlog are assumed to be zero unless otherwise specified in the dataset.
- The Big-M value should be set large enough to not artificially constrain production, but not so large as to cause numerical instability in the solver.
