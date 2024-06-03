
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import time

print('Packages are loaded')

file_name = 'PBAS - Data Case AH.xlsx'


# ### Import data

nr_stores = 49
ti = 0.5 #time interval per hour, each interval is 30 min or 60*ti
times = np.arange(4, 20, ti)

# Load locations and calculate distance

df_dc_location = pd.read_excel(file_name, sheet_name="Store General", header=None).iloc[1:49]

df_stores_location = pd.read_excel(file_name, sheet_name="Store General", header=None).iloc[1:(1+nr_stores), 0:49] ###### can change store nr here
df_merged_locations = pd.concat([df_dc_location, df_stores_location], ignore_index=True, sort=False)

locations = df_merged_locations[3].unique()
num_locations = len(locations)
print(locations)


dist_s = pd.read_excel(file_name, sheet_name="Store General") [["Store nr", "Distance to DC (km)"]].iloc[0:49]
tau_s  = pd.read_excel(file_name, sheet_name="Store General") [["Store nr", "Driving time"]].iloc[0:49]


# load store information
opentime_s = pd.read_excel(file_name, sheet_name="Store General")[["Store nr", "Open"]].iloc[0:49]
closetime_s = pd.read_excel(file_name, sheet_name="Store General")[["Store nr", "Close"]].iloc[0:49]
#print(opentime_s)

demand_wsp = pd.read_excel(file_name, sheet_name="Volume per store per day")[["Day of week", "Store", "Product Group", "Total demand for this day"]]
print(demand_wsp.head())

# truck acces
trucklim_s = pd.read_excel(file_name, sheet_name="Store General")[["Store nr", "Max. allowed truck type"]].iloc[0:49]

truck_types = ['Small', 'Rigid', 'City', 'Euro', 'Electric Small', 'Electric Big']
# Create an empty dictionary to store results
store_truck_dict = {}

for _, row in trucklim_s.iterrows():
    max_truck_index = truck_types.index(row['Max. allowed truck type'])
    for truck in truck_types:
        current_truck_index = truck_types.index(truck)
        # Determine the binary value: 1 if the current truck comes before or is the max allowed truck, else 0
        value = 1 if current_truck_index <= max_truck_index else 0
        # Add to dictionary with (store nr, truck type) as key
        store_truck_dict[(row['Store nr'], truck)] = value  


### Truck information
truck_capacity = pd.read_excel(file_name, sheet_name="Truck types")[["Trucktype", "Capacity Ambient", "Capacity Fresh"]].iloc[:6]
truck_capacity_dict = {}

for _, row in truck_capacity.iterrows():
    truck_capacity_dict[(row['Trucktype'], 'ambient')] = row['Capacity Ambient']
    truck_capacity_dict[(row['Trucktype'], 'fresh')] = row['Capacity Fresh']


truck_cost_km = pd.read_excel(file_name, sheet_name="Truck types")[["Trucktype", "Cost per km"]].iloc[:6]
truck_cost_h = pd.read_excel(file_name, sheet_name="Truck types")[["Trucktype", "Cost per hour"]].iloc[:6]
truck_emission = pd.read_excel(file_name, sheet_name="Truck types")[["Trucktype", "kg CO2 emission per km"]].iloc[:6]
truck_range = pd.read_excel(file_name, sheet_name="Truck types")[["Trucktype", "Range"]].iloc[:6]

print(truck_range)
#print(truck_cost_h)


# ### Model and constraints



#%%

# Create a Gurobi model
model = gp.Model("PBAS_AH")

# Create sets
#locs = locations.tolist()
#D = [locs[0]]

S = locations
print(S)


#D_S = D + S

V = truck_types
W = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
P = ["ambient", "fresh"]
Q = [1, 2]
T = times.tolist()


for s in S:
    store_truck_dict[(s, 'Electric Small')] = 1
    store_truck_dict[(s, 'Electric Big')] = store_truck_dict[(s, 'City')]    
            
# Print the resulting dictionary

#print(store_truck_dict)


# Create parameters
#dist_ij = distances_dict
#tau_ij  = time_dict

open_s  = pd.Series(opentime_s['Open'].values, index=opentime_s['Store nr']).to_dict()
close_s = pd.Series(closetime_s['Close'].values, index= closetime_s['Store nr']).to_dict()
delta_wsp = pd.Series(demand_wsp["Total demand for this day"].values, index=pd.MultiIndex.from_frame(demand_wsp[["Day of week", "Store", "Product Group", ]])).to_dict()

PHI_sv  = store_truck_dict

cap_vp  = truck_capacity_dict
Cdist_v = pd.Series(truck_cost_km['Cost per km'].values, index=truck_cost_km['Trucktype']).to_dict()
Chour_v = pd.Series(truck_cost_h['Cost per hour'].values, index=truck_cost_h['Trucktype']).to_dict()
E_v     = pd.Series(truck_emission["kg CO2 emission per km"].values, index=truck_emission['Trucktype']).to_dict()
ran_v   = pd.Series(truck_range["Range"].values, index=truck_range['Trucktype']).to_dict()

lt = loadingtime
ult = unloadingtime
iat = interarrivaltime
Elim = Emission_lim
ipt = Irregularity_pen

dist_store_s  = pd.Series(dist_s['Distance to DC (km)'].values, index=dist_s['Store nr']).to_dict()
tau_store_s = pd.Series(tau_s['Driving time'].values, index= tau_s['Store nr']).to_dict()

print(Cdist_v)

print(cap_vp)
#print(PHI_sv)


# Create decision variables
#x_wtpq      = model.addVars(W, T, P, Q, vtype=GRB.BINARY, name="x")      # truck is loaded at wtpq
y_wtpqs     = model.addVars(W, T, P, Q, S, vtype=GRB.BINARY, name="y")      # truck loaded at wtpq goes to s
phi_wtpqv   = model.addVars(W, T, P, Q, V, vtype=GRB.BINARY, name="phi")    # truck loaded at wtpq is of type v
z_wtpqsv    = model.addVars(W, T, P, Q, S, V, vtype=GRB.BINARY, name="z")      # truck loaded at wtpq goes to s and is of type v

#Tarr_wpqs  = model.addVars(W, P, Q, S, vtype=GRB.CONTINUOUS, name="Tarr", lb=0.0, ub=24)      # time truck wtpqs arrives at s
#CH_cost  = model.addVars(vtype=GRB.CONTINUOUS, name="CH", lb=0.0)      # hour cost of truck wtpq
#CKM_cost  = model.addVars(vtype=GRB.CONTINUOUS, name="CKM", lb=0.0)      # km cost of truck wtpq
#Emission = model.addVars(vtype=GRB.CONTINUOUS, name="CKM", lb=0.0)

I_wsp       = model.addVars(W, S, P, vtype=GRB.INTEGER, name="I", lb=0.0, ub= 250)       # Inventory store s
R_wtpqs     = model.addVars(W, T, P, Q, S, vtype=GRB.INTEGER, name="R", lb=0.0, ub=70)   # replenishment of truck to store s


#### Constraints ####
### ASSIGNMENT AND ROUTING CONSTRAINTS ###
# can only visit one store


model.addConstrs(
    (gp.quicksum(y_wtpqs[w, t, p, q, s] for s in S) <= 1
     for w in W for t in T for p in P for q in Q),
    name="Constraint_1"
)
                           
model.addConstrs(
    (z_wtpqsv[w, t, p, q, s, v] * dist_store_s[s] * 2 <= ran_v[v]
     for w in W for t in T for p in P for q in Q for s in S for v in V),
    name="Constraint_4.1"
)

model.addConstrs(
    (z_wtpqsv[w, t, p, q, s, v] <=  PHI_sv[s, v] 
     for w in W for t in T for p in P for q in Q for s in S for v in V),
    name="Constraint_3"
)

model.addConstrs(
    (z_wtpqsv[w, t, p, q, s, v] == y_wtpqs[w, t, p, q, s] * phi_wtpqv[w, t, p, q, v]
     for w in W for t in T for p in P for q in Q for s in S for v in V),
    name="Constraint_4"
)

model.addConstrs( 
    ((t + 0.5 + tau_store_s[s]) >= open_s[s] *  y_wtpqs[w, t, p, q, s]
        for w in W for t in T for p in P for q in Q for s in S),
    name="Constraint_8"
)

model.addConstrs(
    ((t + 0.5 + tau_store_s[s] + 0.5) *  y_wtpqs[w, t, p, q, s] <= close_s[s] 
        for w in W for t in T for p in P for q in Q for s in S),
    name="Constraint_9"
)

# Constraint nr 15
model.addConstrs(
    (I_wsp[w, s, p] >= delta_wsp[w, s, p] 
     for w in W for s in S for p in P),
    name="Constraint_18"
)

# Constraint nr 10
model.addConstrs(
    (I_wsp[w, s, p] == gp.quicksum(z_wtpqsv[w, t, p, q, s, v] * cap_vp[v, p] for t in T for q in Q for v in V) 
     for w in W for s in S for p in P),
    name="Constraint_13"
)


CH_cost = gp.quicksum(z_wtpqsv[w, t, p, q, s, v] * Chour_v[v] * (lt + ult + 2*tau_store_s[s]) 
                      for s in S for v in V for w in W for t in T for p in P for q in Q)
                    
CKM_cost = gp.quicksum(z_wtpqsv[w, t, p, q, s, v] * Cdist_v[v] * (2* dist_store_s[s]) 
                       for s in S for v in V for w in W for t in T for p in P for q in Q)

Emission = gp.quicksum(z_wtpqsv[w, t, p, q, s, v] * E_v[v] * (2* dist_store_s[s]) 
                       for s in S for v in V for w in W for t in T for p in P for q in Q)


gp.setParam('OutputFlag', 1)
gp.setParam('LogFile', 'gurobi.log')


#%%

# Objective function
obj = CH_cost + CKM_cost + Emission 
   


#%%


# Optimize the model
model.setObjective(obj, GRB.MINIMIZE)
# model.setParam('TimeLimit', 40) 
model.setParam('Cuts', 3)          # More aggressive cut generation
model.setParam('MIPFocus', 1)      # Focus on finding good feasible solutions
model.optimize()

# Check the optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
elif model.status == GRB.INFEASIBLE:
    print("The model is infeasible.")
elif model.status == GRB.UNBOUNDED:
    print("The model is unbounded.")
else:
    print("Optimization was stopped with status", model.status)
et=time.time()


print("The Cost is:", model.objVal) 


# Check if the model is infeasible
if model.status == gp.GRB.Status.INFEASIBLE:
    print("Model is infeasible.")

    # Compute and print the Irreducible Inconsistent Subsystem (IIS)
    model.computeIIS()
    print("\nConstraints causing infeasibility:")
    for constr in model.getConstrs():
        if constr.IISConstr:
            print(constr.constrName)
else:
    print("Model is feasible.")


print(f"day, product, q |  t  |  store | truck |delivery|demand|")
print("--------------------------------------------------------")
for w in W:
    for p in P:
        for s in S:
            # Iterate over other variables while keeping w, p, and s constant
            for q in Q:
                for t in T:
                    for v in V:
                        if z_wtpqsv[w, t, p, q, s, v].x == 1:
                            print(f"{w}, {p}, {q} | {t} | {s} | {v} | {I_wsp[w,s,p].x} | {delta_wsp[w,s,p]}")                          


final_CH_cost = sum(z_wtpqsv[w, t, p, q, s, v].x * Chour_v[v] * (lt + ult + 2 * tau_store_s[s]) 
                    for s in S for v in V for w in W for t in T for p in P for q in Q)
    
final_CKM_cost = sum(z_wtpqsv[w, t, p, q, s, v].x * Cdist_v[v] * (2 * dist_store_s[s]) 
                     for s in S for v in V for w in W for t in T for p in P for q in Q)
    
final_Emission = sum(z_wtpqsv[w, t, p, q, s, v].x * E_v[v] * (2 * dist_store_s[s]) 
                     for s in S for v in V for w in W for t in T for p in P for q in Q)

print("Distance Cost:", final_CKM_cost)
print("Hour Cost:", final_CH_cost)
print ("CO2 Emission:",final_Emission)


# Initialize count

count_v_new = 0
count_v = 0

# Iterate over all combinations of indices w, t, p, q, s
for w in W:
    for t in T:
        for p in P:
            for q in Q:
                for s in S:
                    for v in ['Electric Small', 'Electric Big']:
                        # Check if z_wtpqsv[w, t, p, q, s, v] equals 5 for any v
                        if z_wtpqsv[w, t, p, q, s, v].x == 1:
                            count_v_new += 1
                    for v in V:
                        if z_wtpqsv[w, t, p, q, s, v].x == 1:
                            count_v += 1
                            


print("Number of times v_new:", count_v_new)
print("Number of times v:", count_v)
print("New truck is:", count_v_new/count_v)



