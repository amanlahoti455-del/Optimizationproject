import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import pandas as pd
import pulp as pl

# Updating dataset paths
PRODUCTS_CSV = "01_products_6m_B.csv.csv"
DEMAND_CSV   = "02_demand_6m_B.csv.csv"
CAPACITY_CSV = "03_capacity_6m_B.csv.csv"


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#loading data and cleaning column names
products_df = pd.read_csv(PRODUCTS_CSV)
demand_df   = pd.read_csv(DEMAND_CSV)
capacity_df = pd.read_csv(CAPACITY_CSV)

# removing BOMs or whitespace from column names
products_df.columns = products_df.columns.str.strip().str.replace('\ufeff', '')
demand_df.columns   = demand_df.columns.str.strip().str.replace('\ufeff', '')
capacity_df.columns = capacity_df.columns.str.strip().str.replace('\ufeff', '')

# preserve month labels exactly as in CSV (e.g., 'Jul', 'Aug' or 1,2,..)
months = demand_df.iloc[:, 0].tolist()

# 2) PARAMS / DICTS
products = products_df["Product"].astype(str).tolist()

prod_cost = dict(zip(products_df.Product, products_df.ProdCost.astype(float)))
inv_cost  = dict(zip(products_df.Product, products_df.CInv.astype(float)))
sell_price= dict(zip(products_df.Product, products_df.SellPrice.astype(float)))
rres      = dict(zip(products_df.Product, products_df.RRes.astype(float)))

# checks if backlog cost is correctly used as 2*(Cost of product + inventory cost)
back_cost = {}
for _, row in products_df.iterrows():
    p = row["Product"]
    if "CBack" in products_df.columns and not pd.isna(row.get("CBack", None)) and row.get("CBack", 0) > 0:
        back_cost[p] = float(row["CBack"])
    else:
        back_cost[p] = 2.0 * (float(row["ProdCost"]) + float(row["CInv"]))

# building a nested dictionary for demand of a particular product in a particular month
demand = {}
for _, row in demand_df.iterrows():
    m = row.iloc[0]
    demand[m] = {p: int(row[p]) for p in products}

# capacity mapping: capacity[month_label] -> Lt 
cap_months = capacity_df.iloc[:, 0].tolist()
cap_values = capacity_df.iloc[:, 1].tolist()
capacity = {cap_months[i]: float(cap_values[i]) for i in range(len(cap_months))}

# initial inventory is considered to be 0 for all products
initial_inventory = {p: 0 for p in products}

# 3)Intialising CSetup and Big-M
avg_prod_cost = products_df["ProdCost"].astype(float).mean()
CSetup = float(10.0 * avg_prod_cost)

# Product-specific Big-M per month: maximum units of product p producible in month m
M_dict = {}
for m in months:
    Lt = capacity.get(m, None)
    if Lt is None:
        Lt = list(capacity.values())[months.index(m)]
    for p in products:
        M_dict[(p, m)] = max(0, math.floor(float(Lt) / float(rres[p])))

# Checking Feasibility (sum of (demand* rres for product)< Lt)
print("CAPACITY FEASIBILITY CHECK")
feasible_all = True
for m in months:
    Lt = capacity.get(m, None)
    if Lt is None:
        Lt = list(capacity.values())[months.index(m)]
    required_hours = sum(demand[m][p] * rres[p] for p in products)
    feasible = (required_hours <= Lt)
    print(f"Month '{m}': required_hours={required_hours:.1f}, capacity={Lt:.1f}, feasible={feasible}")
    if not feasible:
        feasible_all = False

if not feasible_all:
    print("\nOne or more months have demand that exceeds capacity. The model will produce up to capacity and incur backlog for unavoidable shortage.")
else:
    print("\nAll months feasible: capacity >= required hours.")

# MODEL
model = pl.LpProblem("Production_Planning", pl.LpMinimize)

# Decision variables
P = pl.LpVariable.dicts("P", (products, months), lowBound=0, cat="Continuous")
I = pl.LpVariable.dicts("I", (products, months), lowBound=0, cat="Continuous")
B = pl.LpVariable.dicts("B", (products, months), lowBound=0, cat="Continuous")
Y = pl.LpVariable.dicts("Y", months, cat="Binary")

# Objective function
model += (
    pl.lpSum(prod_cost[p] * P[p][m] for p in products for m in months)
    + pl.lpSum(inv_cost[p]  * I[p][m] for p in products for m in months)
    + pl.lpSum(back_cost[p] * B[p][m] for p in products for m in months)
    + pl.lpSum(CSetup * Y[m] for m in months)
), "Total_Cost"

#CONSTRAINTS
#1. Inventory balance
for p in products:
    for idx, m in enumerate(months):
        if idx == 0:
            model += (
                initial_inventory[p] + P[p][m] - demand[m][p] == I[p][m] - B[p][m]
            ), f"InvBal_{p}_{m}"
        else:
            prev_m = months[idx - 1]
            model += (
                I[p][prev_m] + P[p][m] - demand[m][p] == I[p][m] - B[p][m]
            ), f"InvBal_{p}_{m}"

#2. Capacity (resource hours)
for m in months:
    Lt = capacity.get(m, None)
    if Lt is None:
        Lt = list(capacity.values())[months.index(m)]
    model += (
        pl.lpSum(rres[p] * P[p][m] for p in products) <= Lt
    ), f"Cap_{m}"

# 3. Ensures that backlog can not be more than demand
for p in products:
    for m in months:
        model += (
            B[p][m] <= demand[m][p]
        ), f"Backlog_upper_{p}_{m}"

# 4. Setup linking tightened per product
for p in products:
    for m in months:
        model += (
            P[p][m] <= M_dict[(p, m)] * Y[m]
        ), f"SetupLink_{p}_{m}"

# 5 Explicit non-negativity 
for p in products:
    for m in months:
        model += P[p][m] >= 0
        model += I[p][m] >= 0
        model += B[p][m] >= 0

# Solving using the CBC solver
solver = pl.PULP_CBC_CMD(msg=False, timeLimit=300)
result = model.solve(solver)

print("SOLVER STATUS")
print(pl.LpStatus[model.status])
print("Objective value (total cost):", float(pl.value(model.objective)))
print("\n")

# Exporting results
prod_rows = []
inv_rows  = []
back_rows = []
for m in months:
    for p in products:
        prod_val = 0 if P[p][m].value() is None else P[p][m].value()
        inv_val  = 0 if I[p][m].value() is None else I[p][m].value()
        back_val = 0 if B[p][m].value() is None else B[p][m].value()
        # round to nearest integer for units
        prod_rows.append((m, p, int(round(prod_val))))
        inv_rows.append((m, p, int(round(inv_val))))
        back_rows.append((m, p, int(round(back_val))))

pd.DataFrame(prod_rows, columns=["Month", "Product", "Production"]).to_csv(
    os.path.join(OUTPUT_DIR, "results_production.csv"), index=False)
pd.DataFrame(inv_rows,  columns=["Month", "Product", "Inventory"]).to_csv(
    os.path.join(OUTPUT_DIR, "results_inventory.csv"), index=False)
pd.DataFrame(back_rows, columns=["Month", "Product", "Backlog"]).to_csv(
    os.path.join(OUTPUT_DIR, "results_backlog.csv"), index=False)
pd.DataFrame([(m, int(Y[m].value() or 0)) for m in months], columns=["Month", "Setup"]).to_csv(
    os.path.join(OUTPUT_DIR, "results_setup.csv"), index=False)

# printing cost breakdown
total_prod_cost = sum(prod_cost[p] * sum(r for (mm, pp, r) in prod_rows if pp == p) for p in products)
total_inv_cost  = sum(inv_cost[p]  * sum(r for (mm, pp, r) in inv_rows  if pp == p) for p in products)
total_back_cost = sum(back_cost[p] * sum(r for (mm, pp, r) in back_rows if pp == p) for p in products)
total_setup_cost = CSetup * sum(int(Y[m].value() or 0) for m in months)

print("Cost breakdown:",
      f"Prod={total_prod_cost:.2f}, Inv={total_inv_cost:.2f}, Back={total_back_cost:.2f}, Setup={total_setup_cost:.2f}")

print("Outputs saved to:", OUTPUT_DIR)