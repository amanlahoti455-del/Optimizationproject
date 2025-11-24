import os
import math
import copy
import pandas as pd
import pulp as pl
import matplotlib.pyplot as plt
import seaborn as sns

# PATH FIX — WORKS ON GITHUB & LOCALLY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
DATA_DIR = os.path.join(BASE_DIR, "..", "Datasets")

PRODUCTS_CSV = os.path.join(DATA_DIR, "01_products.csv")
DEMAND_CSV   = os.path.join(DATA_DIR, "02_demand.csv")
CAPACITY_CSV = os.path.join(DATA_DIR, "03_capacity.csv")

SOLVER = pl.PULP_CBC_CMD(msg=False, timeLimit=300)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# LOAD DATA
def load_data():
    products_df = pd.read_csv(PRODUCTS_CSV)
    demand_df   = pd.read_csv(DEMAND_CSV)
    capacity_df = pd.read_csv(CAPACITY_CSV)

    # Clean column names
    products_df.columns = products_df.columns.str.strip().str.replace('\ufeff', '')
    demand_df.columns   = demand_df.columns.str.strip().str.replace('\ufeff', '')
    capacity_df.columns = capacity_df.columns.str.strip().str.replace('\ufeff', '')

    months = demand_df.iloc[:, 0].tolist()
    products = products_df["Product"].astype(str).tolist()

    prod_cost = dict(zip(products_df.Product, products_df.ProdCost.astype(float)))
    inv_cost  = dict(zip(products_df.Product, products_df.CInv.astype(float)))
    sell_price= dict(zip(products_df.Product, products_df.SellPrice.astype(float)))
    rres      = dict(zip(products_df.Product, products_df.RRes.astype(float)))

    # Backlog cost auto-handling
    back_cost = {}
    for _, row in products_df.iterrows():
        p = row["Product"]
        if "CBack" in products_df.columns and not pd.isna(row["CBack"]) and row["CBack"] > 0:
            back_cost[p] = float(row["CBack"])
        else:
            back_cost[p] = 2.0 * (row["ProdCost"] + row["CInv"])

    demand = {row.iloc[0]: {p: int(row[p]) for p in products} 
              for _, row in demand_df.iterrows()}

    capacity = {capacity_df.iloc[i, 0]: float(capacity_df.iloc[i,1])
                for i in range(len(capacity_df))}

    return {
        "products_df": products_df,
        "demand_df": demand_df,
        "capacity_df": capacity_df,
        "months": months,
        "products": products,
        "prod_cost": prod_cost,
        "inv_cost": inv_cost,
        "sell_price": sell_price,
        "rres": rres,
        "back_cost": back_cost,
        "demand": demand,
        "capacity": capacity
    }


# BUILD & SOLVE MODEL

def build_and_solve(data, CSetup=None, solver=SOLVER):
    products  = data["products"]
    months    = data["months"]
    prod_cost = data["prod_cost"]
    inv_cost  = data["inv_cost"]
    back_cost = data["back_cost"]
    rres      = data["rres"]
    demand    = data["demand"]
    capacity  = data["capacity"]
    products_df = data["products_df"]

    # Default setup cost
    if CSetup is None:
        CSetup = 10.0 * products_df["ProdCost"].astype(float).mean()

    # Big-M values
    M_dict = {}
    for m in months:
        for p in products:
            M_dict[(p, m)] = max(0, math.floor(capacity[m] / rres[p]))

    model = pl.LpProblem("Production_Planning", pl.LpMinimize)

    # Decision variables
    P = pl.LpVariable.dicts("P", (products, months), lowBound=0)
    I = pl.LpVariable.dicts("I", (products, months), lowBound=0)
    B = pl.LpVariable.dicts("B", (products, months), lowBound=0)
    Y = pl.LpVariable.dicts("Y", months, cat="Binary")

    # Objective
    model += (
        pl.lpSum(prod_cost[p] * P[p][m] for p in products for m in months) +
        pl.lpSum(inv_cost[p]  * I[p][m] for p in products for m in months) +
        pl.lpSum(back_cost[p] * B[p][m] for p in products for m in months) +
        pl.lpSum(CSetup * Y[m] for m in months)
    )

    # Inventory Balance
    for p in products:
        for idx, m in enumerate(months):
            if idx == 0:
                model += (P[p][m] - demand[m][p] == I[p][m] - B[p][m])
            else:
                prev = months[idx - 1]
                model += (I[p][prev] + P[p][m] - demand[m][p] == I[p][m] - B[p][m])

    # Capacity
    for m in months:
        model += (pl.lpSum(rres[p] * P[p][m] for p in products) <= capacity[m])

    # Backlog + Setup Linking
    for p in products:
        for m in months:
            model += (B[p][m] <= demand[m][p])
            model += (P[p][m] <= M_dict[(p, m)] * Y[m])

    # Solve
    model.solve(solver)

    # Collect Outputs
    prod_rows, inv_rows, back_rows, setup_rows = [], [], [], []
    for m in months:
        for p in products:
            prod_rows.append((m, p, int(round(P[p][m].value() or 0))))
            inv_rows.append((m, p, int(round(I[p][m].value() or 0))))
            back_rows.append((m, p, int(round(B[p][m].value() or 0))))
        setup_rows.append((m, int(Y[m].value() or 0)))

    total_prod_cost = sum(prod_cost[p] * sum(r for (_,pp,r) in prod_rows if pp == p) for p in products)
    total_inv_cost  = sum(inv_cost[p]  * sum(r for (_,pp,r) in inv_rows  if pp == p) for p in products)
    total_back_cost = sum(back_cost[p] * sum(r for (_,pp,r) in back_rows if pp == p) for p in products)
    total_setup_cost = CSetup * sum(v for (_,v) in setup_rows)

    cost_breakdown = {
        "Status": pl.LpStatus[model.status],
        "Objective": float(pl.value(model.objective)),
        "ProdCost": total_prod_cost,
        "InvCost": total_inv_cost,
        "BackCost": total_back_cost,
        "SetupCost": total_setup_cost
    }

    return {
        "production_rows": prod_rows,
        "inventory_rows": inv_rows,
        "backlog_rows": back_rows,
        "setup_rows": setup_rows,
        "cost_breakdown": cost_breakdown
    }

# PLOTTING FUNCTIONS

def plot_scenario(outputs, data, scenario_name):
    products = data["products"]
    months   = data["months"]
    rres     = data["rres"]
    capacity = data["capacity"]

    prod_df = pd.DataFrame(outputs["production_rows"], 
                           columns=["Month","Product","Production"])
    inv_df  = pd.DataFrame(outputs["inventory_rows"], 
                           columns=["Month","Product","Inventory"])
    back_df = pd.DataFrame(outputs["backlog_rows"], 
                           columns=["Month","Product","Backlog"])

    prod_pivot = prod_df.pivot(index="Month", columns="Product", values="Production").fillna(0)
    inv_pivot  = inv_df.pivot(index="Month", columns="Product", values="Inventory").fillna(0)
    back_pivot = back_df.pivot(index="Month", columns="Product", values="Backlog").fillna(0)

    cb = outputs["cost_breakdown"]

    fig = plt.figure(figsize=(14,10))
    gs = fig.add_gridspec(2,2)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.pie(
        [cb["ProdCost"], cb["InvCost"], cb["BackCost"], cb["SetupCost"]],
        labels=["Production","Inventory","Backlog","Setup"],
        autopct="%1.1f%%"
    )
    ax0.set_title(f"{scenario_name}: Cost Breakdown")

    ax1 = fig.add_subplot(gs[0,1])
    prod_pivot.plot(kind="bar", ax=ax1)
    ax1.set_title("Production Schedule")

    ax2 = fig.add_subplot(gs[1,0])
    inv_pivot.plot(ax=ax2, marker="o")
    ax2.set_title("Inventory Levels")

    ax3 = fig.add_subplot(gs[1,1])
    back_pivot.plot(kind="area", ax=ax3, alpha=0.6)
    ax3.set_title("Backlog")

    plt.tight_layout()
    plt.show()

    # Capacity Usage Graph
    used_hours = []
    for m in months:
        used = sum(
            rres[p] *
            prod_df[(prod_df["Month"] == m) & (prod_df["Product"] == p)]["Production"].iloc[0]
            for p in products
        )
        used_hours.append(used)

    cap_df = pd.DataFrame({
        "Month": months,
        "Used": used_hours,
        "Available": [capacity[m] for m in months]
    })

    plt.figure(figsize=(10,5))
    plt.bar(cap_df["Month"], cap_df["Used"], label="Used")
    plt.bar(
        cap_df["Month"], cap_df["Available"] - cap_df["Used"],
        bottom=cap_df["Used"], alpha=0.3, label="Unused"
    )
    plt.title(f"{scenario_name}: Capacity Utilization")
    plt.legend()
    plt.show()

    print(f"{scenario_name} SUMMARY")
    print("Status:", cb["Status"])
    print("Objective:", cb["Objective"])
    print()


# SCENARIO FUNCTIONS

def run_and_plot_scenarios():
    data_orig = load_data()

    scenarios = [
        ("base", lambda d: d),
        ("demand_plus20", lambda d: modify_demand(d, 1.20)),
        ("demand_minus20", lambda d: modify_demand(d, 0.80)),
        ("capacity_plus15", lambda d: modify_capacity(d, 1.15)),
        ("capacity_minus15", lambda d: modify_capacity(d, 0.85)),
        ("cost_prod_plus10", lambda d: modify_prod_cost(d, 1.10)),
        ("rres_plus10", lambda d: modify_rres(d, 1.10)),
        ("rres_minus10", lambda d: modify_rres(d, 0.90)),
        ("demand+20_cap-15", lambda d: modify_capacity(modify_demand(d,1.20), 0.85))
    ]

    for name, modify in scenarios:
        print(f"\n Scenario: {name} ")
        dcopy = copy_data_for_scenario(data_orig)
        newd = modify(dcopy)
        outputs = build_and_solve(newd)
        plot_scenario(outputs, newd, name)


# Helper Modify Functions

def copy_data_for_scenario(original):
    return {key: (original[key].copy() if hasattr(original[key], "copy") else original[key])
            for key in original}

def modify_demand(data, scale):
    for m in data["months"]:
        for p in data["products"]:
            data["demand"][m][p] = int(round(data["demand"][m][p] * scale))
    return data

def modify_capacity(data, scale):
    for m in data["months"]:
        data["capacity"][m] = float(round(data["capacity"][m] * scale, 2))
    return data

def modify_prod_cost(data, scale):
    for p in data["products"]:
        data["prod_cost"][p] = float(round(data["prod_cost"][p] * scale, 2))
    return data

def modify_rres(data, scale):
    for p in data["products"]:
        data["rres"][p] = float(round(data["rres"][p] * scale, 4))
    return data


#DEMO FUNCTION (can change parameters to see how output changes)

def run_live_demo(month, 
                  d_tshirt=None, d_jeans=None, d_hoodie=None, d_jacket=None,
                  cp_tshirt=None, cp_jeans=None, cp_hoodie=None, cp_jacket=None,
                  sp_tshirt=None, sp_jeans=None, sp_hoodie=None, sp_jacket=None,
                  new_capacity=None, new_csetup=None):

    data = load_data()

    if month not in data["months"]:
        print(f"Month '{month}' not found. Available:", data["months"])
        return

    # Update demand
    if d_tshirt is not None: data["demand"][month]["P1_TShirt"] = d_tshirt
    if d_jeans  is not None: data["demand"][month]["P2_Jeans"]  = d_jeans
    if d_hoodie is not None: data["demand"][month]["P3_Hoodie"] = d_hoodie
    if d_jacket is not None: data["demand"][month]["P4_Jacket"] = d_jacket

    # Update Cost Price (ProdCost)
    if cp_tshirt is not None: data["prod_cost"]["P1_TShirt"] = cp_tshirt
    if cp_jeans  is not None: data["prod_cost"]["P2_Jeans"]  = cp_jeans
    if cp_hoodie is not None: data["prod_cost"]["P3_Hoodie"] = cp_hoodie
    if cp_jacket is not None: data["prod_cost"]["P4_Jacket"] = cp_jacket

    # Update Selling Price (SP)
    if sp_tshirt is not None: data["sell_price"]["P1_TShirt"] = sp_tshirt
    if sp_jeans  is not None: data["sell_price"]["P2_Jeans"]  = sp_jeans
    if sp_hoodie is not None: data["sell_price"]["P3_Hoodie"] = sp_hoodie
    if sp_jacket is not None: data["sell_price"]["P4_Jacket"] = sp_jacket

    # Auto update CInv and CBack
    for p in data["products"]:
        cprod = data["prod_cost"][p]
        data["inv_cost"][p] = 0.10 * cprod
        data["back_cost"][p] = 2 * (cprod + data["inv_cost"][p])

        data["products_df"].loc[data["products_df"]["Product"] == p, "ProdCost"] = cprod
        data["products_df"].loc[data["products_df"]["Product"] == p, "CInv"] = data["inv_cost"][p]
        data["products_df"].loc[data["products_df"]["Product"] == p, "CBack"] = data["back_cost"][p]

    # Capacity update
    if new_capacity is not None:
        data["capacity"][month] = new_capacity

    # Setup cost update
    if new_csetup is None:
        new_csetup = float(10 * data["products_df"]["ProdCost"].mean())

    # Solve model
    outputs = build_and_solve(data, CSetup=new_csetup)

    # Extract month results
    prod = {p:q for (m,p,q) in outputs["production_rows"] if m == month}
    inv  = {p:q for (m,p,q) in outputs["inventory_rows"] if m == month}
    back = {p:q for (m,p,q) in outputs["backlog_rows"]    if m == month}

    used = sum(prod.get(p,0) * data["rres"][p] for p in data["products"])
    avail= data["capacity"][month]

    print("SHORT RESULT\n")
    print(f"Month Modified: {month}")
    print(f"Total Cost: {outputs['cost_breakdown']['Objective']:.2f}")
    print(f"Status: {outputs['cost_breakdown']['Status']}\n")

    print("What Was Changed")
    if any(v is not None for v in [d_tshirt, d_jeans, d_hoodie, d_jacket]):
        print("• Demand updated")
    if any(v is not None for v in [cp_tshirt, cp_jeans, cp_hoodie, cp_jacket]):
        print("• Cost Price updated")
    if any(v is not None for v in [sp_tshirt, sp_jeans, sp_hoodie, sp_jacket]):
        print("• Selling Price updated")
    if new_capacity is not None:
        print("• Capacity updated")
    if new_csetup is not None:
        print("• Setup cost updated")

    print("\nModel Decisions")
    print("Production:", prod)
    print("Inventory:", inv)
    print("Backlog:", back)

    print("\nCapacity Usage")
    print(f"Used: {used:.2f}")
    print(f"Available: {avail}")
    print(f"Utilization: {used/avail*100:.1f}%")

    return {
        "Production": prod,
        "Inventory": inv,
        "Backlog": back,
        "TotalCost": outputs["cost_breakdown"]["Objective"]
    }


# MAIN
if __name__ == "__main__":
    run_and_plot_scenarios()
