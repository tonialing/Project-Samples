import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data from Excel file
data_file = 'case_2_value_data.xlsx'
data_file = data_file.replace('\\', '/')
demand_df = pd.read_excel(data_file, sheet_name='Demand_N')
demand_data = {index: row['Demand'] for index, row in demand_df.iterrows()}

# also load this one if you want to simulate data form student-T distribution
demand_studentT_df = pd.read_excel(data_file, sheet_name='Demand_T_student')
demand_studentT_data = {index: row['Demand'] for index, row in demand_studentT_df.iterrows()}

# assume 100 simulations 
num_iterations = 100  

def simulate_network(data, num_iterations=100):
    
    # echelon back-stock level
    Back_levels = {'C': 51.74, 'D': 117.07, 'S': 163.32}

    # initialize variables
    total_cost = 0
    expected_costs = 0
    
    total_demand = 0
    unmet_demand = 0
    stockout_times = 0
    
    max_cost = 0
    min_cost = 0

    Stockout = {t: 0 for t in range(num_iterations + 1)}
    On_hand_inventory = {t: {'C': 31.74, 'D': 5.33, 'S': 6.25} for t in range(num_iterations + 1)}
    In_transit_inventory = {t: {'C': 20, 'D': 20, 'S': 20} for t in range(-2, num_iterations + 1)}
    Ending_level = {t: {'C': 31.74, 'D': 5.33, 'S': 6.25} for t in range(num_iterations + 1)}
    Cost = {t: 0 for t in range(num_iterations + 1)}
    
    for t in range(1, num_iterations + 1):
    
        # demand occurs
        data[t] = data[t] + Stockout[t - 1]  # Corrected this line
            
        # iteration in echelon C
        On_hand_inventory[t]['C'] = Ending_level[t-1]['C'] + In_transit_inventory[t - 1]['C']
            
        #stockout
        Stockout[t] = max(0,  data[t] - On_hand_inventory[t]['C'])  # Corrected this line
            
        In_transit_inventory[t]['C'] = max(0, min((Ending_level[t-1]['D'] + In_transit_inventory[t - 3]['D']), Back_levels['C'] - On_hand_inventory[t]['C'] + Stockout[t-1]))

        Ending_level[t]['C'] = max(0, On_hand_inventory[t]['C']  - data[t])  # Corrected this line       
            
        # iteration in echelon D
        On_hand_inventory[t]['D'] = Ending_level[t-1]['D'] + In_transit_inventory[t - 3]['D']

        In_transit_inventory[t]['D'] = max(0, min((Ending_level[t-1]['S'] + In_transit_inventory[t - 2]['S']), 
                                         Back_levels['D'] - On_hand_inventory[t]['C'] - On_hand_inventory[t]['D'] - In_transit_inventory[t-1]['D']
                                           - In_transit_inventory[t-2]['D'] + Stockout[t-1]))

        Ending_level[t]['D'] = max(0, On_hand_inventory[t]['D'] - In_transit_inventory[t]['C'])
        
        # iteration in echelon S
        On_hand_inventory[t]['S'] = Ending_level[t-1]['S'] + In_transit_inventory[t - 2]['S']
    
        In_transit_inventory[t]['S'] = min(22, Back_levels['S'] - On_hand_inventory[t]['S'] - In_transit_inventory[t-1]['S']  - On_hand_inventory[t]['D'] - On_hand_inventory[t]['C']
                                    - In_transit_inventory[t-1]['D']- In_transit_inventory[t-2]['D']
                                    + Stockout[t-1] )

        Ending_level[t]['S'] = max(0, On_hand_inventory[t]['S'] -  In_transit_inventory[t]['D'] )
                   
        # local holding cost are 8, 4, 1, and then multiple with those ending inventory level 
        Cost[t] = 8 * Ending_level[t]['C'] + 4 * Ending_level[t]['D'] + 1 * Ending_level[t]['S'] + 40 * Stockout[t]
            
        # update total cost and stockout info
        total_cost += Cost[t]
        total_demand += demand_data[t]

        if Stockout[t] != 0:
            stockout_times += 1
            unmet_demand += Stockout[t]  
        
    # calculate ecpected cost and service levels for each scenario
    expected_cost = total_cost / num_iterations
    type_1_service_level = 1 - (stockout_times / num_iterations)
    type_2_service_level = 1 - (unmet_demand / total_demand)  
   
    # for visualization   
    cost_results = [Cost[t] for t in range(1, num_iterations + 1)]
    iterations = range(1, num_iterations + 1)
    plt.scatter(iterations, cost_results, label='Simulation costs')
    plt.title('Scatter Plot of The Costs')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend({"Normal Distribution":"blue", "Student T Distribution": "yellow"})
  
    # return results
    results = {
        'expected_cost': expected_cost,
        'type_1_service_level': type_1_service_level,
        'type_2_service_level': type_2_service_level,
        'stockout_times': stockout_times,
        #'Stockout': Stockout
        #'Cost': Cost
        }
    return results  

  
# for normal distributions
simulation_results_1 = simulate_network(demand_data)

#print simulation results
print("Simulation Results_1:")
for key, value in simulation_results_1.items():
    print(f"{key}: {value}") 

print ("#######")
