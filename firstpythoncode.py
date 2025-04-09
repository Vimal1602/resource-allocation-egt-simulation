import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import random

class CloudResourceAuction:
    """
    A simulation of resource allocation in cloud computing environments using game theory 
    and auction mechanisms.
    """
    
    def __init__(self, num_vms=5, num_resources=3, resource_capacity=(100, 200, 300)):
        self.num_vms = num_vms  # Number of virtual machines (players)
        self.num_resources = num_resources  # Types of resources (CPU, RAM, Storage)
        self.resource_capacity = resource_capacity  # Total capacity of each resource
        
        # VM resource requirements - randomly generated
        self.vm_requirements = np.random.randint(10, 50, size=(num_vms, num_resources))
        
        # VM valuation of resources - how much each VM values each resource
        self.vm_valuations = np.random.randint(1, 10, size=(num_vms, num_resources))
        
        # Initial bids for each VM for each resource
        self.bids = np.random.randint(1, 5, size=(num_vms, num_resources)).astype(float)
        
        # Allocation matrix - how much of each resource is allocated to each VM
        self.allocations = np.zeros((num_vms, num_resources))
        
        # History for tracking
        self.utility_history = []
        self.bid_history = []
        self.allocation_history = []
    
    def calculate_allocation(self):
        """
        Calculate resource allocation based on proportional share auction mechanism.
        Resources are allocated in proportion to the bids.
        """
        allocation = np.zeros((self.num_vms, self.num_resources))
        
        for r in range(self.num_resources):
            total_bid = np.sum(self.bids[:, r])
            if total_bid > 0:  # Avoid division by zero
                for v in range(self.num_vms):
                    # Proportional allocation: bid_i / total_bids * total_resource
                    proportion = self.bids[v, r] / total_bid if total_bid > 0 else 0
                    allocation[v, r] = proportion * self.resource_capacity[r]
                    
                    # Cap allocation at VM's requirement
                    if allocation[v, r] > self.vm_requirements[v, r]:
                        allocation[v, r] = self.vm_requirements[v, r]
        
        self.allocations = allocation
        return allocation
    
    def calculate_utility(self, vm_idx, bids=None):
        """
        Calculate the utility (payoff) for a specific VM given the current bids.
        Utility = Value of allocated resources - Cost of bids
        """
        if bids is None:
            bids = self.bids
            
        # Save the original bids
        original_bids = self.bids.copy()
        
        # Set the bids to the provided ones temporarily
        self.bids = bids
        
        # Calculate allocations with these bids
        allocation = self.calculate_allocation()
        
        # Calculate utility: value of resources minus cost of bidding
        utility = 0
        for r in range(self.num_resources):
            # Value gained is the product of allocation and valuation
            value_gained = allocation[vm_idx, r] * self.vm_valuations[vm_idx, r]
            # Cost is the bid amount
            bid_cost = bids[vm_idx, r]
            utility += value_gained - bid_cost
        
        # Restore original bids
        self.bids = original_bids
        
        return utility
    
    def optimize_bid(self, vm_idx):
        """
        Find the optimal bid for a VM to maximize its utility,
        given the current bids of other VMs.
        """
        current_bids = self.bids.copy()
        
        # Define the objective function to minimize (negative utility)
        def objective(x):
            new_bids = current_bids.copy()
            new_bids[vm_idx] = x
            return -self.calculate_utility(vm_idx, new_bids)
        
        # Initial guess
        x0 = current_bids[vm_idx]
        
        # Bounds for bids (must be positive but not excessive)
        bounds = [(0.1, 20) for _ in range(self.num_resources)]
        
        # Minimize negative utility
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return result.x
    
    def find_nash_equilibrium(self, iterations=10):
        """
        Iteratively find the Nash Equilibrium by letting each VM optimize its bid
        in response to others until convergence or max iterations.
        """
        for iteration in range(iterations):
            old_bids = self.bids.copy()
            
            # Each VM optimizes its bid in response to others
            for vm_idx in range(self.num_vms):
                optimal_bid = self.optimize_bid(vm_idx)
                self.bids[vm_idx] = optimal_bid
            
            # Calculate current allocations and utilities
            allocation = self.calculate_allocation()
            utilities = [self.calculate_utility(vm_idx) for vm_idx in range(self.num_vms)]
            
            # Store history
            self.bid_history.append(self.bids.copy())
            self.allocation_history.append(allocation.copy())
            self.utility_history.append(utilities)
            
            # Check for convergence
            bid_change = np.abs(self.bids - old_bids).max()
            print(f"Iteration {iteration+1}: Max bid change = {bid_change:.4f}")
            
            if bid_change < 0.01:
                print("Nash Equilibrium reached!")
                break
    
    def plot_results(self):
        """Plot the results of the auction process."""
        # Convert history to arrays
        utility_history = np.array(self.utility_history)
        
        # Plot utilities over iterations
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for vm_idx in range(self.num_vms):
            plt.plot(utility_history[:, vm_idx], label=f'VM {vm_idx+1}')
        plt.title('VM Utilities over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.legend()
        plt.grid(True)
        
        # Plot final resource allocation
        plt.subplot(2, 1, 2)
        allocation_df = pd.DataFrame(self.allocations, 
                              index=[f'VM {i+1}' for i in range(self.num_vms)],
                              columns=[f'Resource {i+1}' for i in range(self.num_resources)])
        allocation_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Final Resource Allocation per VM')
        plt.xlabel('Virtual Machine')
        plt.ylabel('Allocated Amount')
        plt.legend(title='Resource Type')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
        
    def print_detailed_results(self):
        """Print detailed results of the auction."""
        print("\n===== FINAL RESULTS =====")
        print("\nVM Requirements:")
        req_df = pd.DataFrame(self.vm_requirements, 
                             index=[f'VM {i+1}' for i in range(self.num_vms)],
                             columns=[f'Resource {i+1}' for i in range(self.num_resources)])
        print(req_df)
        
        print("\nVM Valuations:")
        val_df = pd.DataFrame(self.vm_valuations, 
                             index=[f'VM {i+1}' for i in range(self.num_vms)],
                             columns=[f'Resource {i+1}' for i in range(self.num_resources)])
        print(val_df)
        
        print("\nFinal Bids:")
        bid_df = pd.DataFrame(self.bids, 
                             index=[f'VM {i+1}' for i in range(self.num_vms)],
                             columns=[f'Resource {i+1}' for i in range(self.num_resources)])
        print(bid_df.round(2))
        
        print("\nFinal Allocations:")
        alloc_df = pd.DataFrame(self.allocations, 
                               index=[f'VM {i+1}' for i in range(self.num_vms)],
                               columns=[f'Resource {i+1}' for i in range(self.num_resources)])
        print(alloc_df.round(2))
        
        print("\nFinal Utilities:")
        utilities = [self.calculate_utility(vm_idx) for vm_idx in range(self.num_vms)]
        util_df = pd.DataFrame({
            'VM': [f'VM {i+1}' for i in range(self.num_vms)],
            'Utility': utilities
        })
        print(util_df.round(2))
        
        # Calculate efficiency metrics
        total_allocated = np.sum(self.allocations, axis=0)
        efficiency = total_allocated / self.resource_capacity
        print("\nResource Utilization Efficiency:")
        eff_df = pd.DataFrame({
            'Resource': [f'Resource {i+1}' for i in range(self.num_resources)],
            'Total Capacity': self.resource_capacity,
            'Total Allocated': total_allocated,
            'Efficiency (%)': efficiency * 100
        })
        print(eff_df.round(2))

# Running the simulation
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    print("===== Cloud Resource Allocation using Game Theory =====")
    print("Simulating resource allocation among VMs using auction mechanisms...")
    
    # Initialize the auction
    auction = CloudResourceAuction(num_vms=5, num_resources=3, resource_capacity=(100, 150, 200))
    
    # Initial state
    print("\nINITIAL STATE:")
    print("VM Requirements:")
    print(pd.DataFrame(auction.vm_requirements, 
                      index=[f'VM {i+1}' for i in range(auction.num_vms)],
                      columns=[f'Resource {i+1}' for i in range(auction.num_resources)]))
    
    # Run the Nash Equilibrium finding process
    print("\nFinding Nash Equilibrium...")
    auction.find_nash_equilibrium(iterations=15)
    
    # Print detailed results
    auction.print_detailed_results()
    
    # Plot results
    auction.plot_results()
    
    print("\nCOMPARISON WITH FIRST-PRICE AUCTION ALLOCATION")
    # Implementing a simple first-price auction for comparison
    def first_price_auction(vm_requirements, resource_capacity, random_bids):
        num_vms, num_resources = vm_requirements.shape
        allocation = np.zeros((num_vms, num_resources))
        
        for r in range(num_resources):
            # Sort VMs by bid for this resource (descending)
            vm_order = np.argsort(-random_bids[:, r])
            remaining = resource_capacity[r]
            
            for vm_idx in vm_order:
                # Allocate either what VM needs or what's remaining
                allocated = min(vm_requirements[vm_idx, r], remaining)
                allocation[vm_idx, r] = allocated
                remaining -= allocated
                if remaining <= 0:
                    break
        
        return allocation
    
    # Generate random bids for first-price auction
    random_bids = np.random.randint(1, 10, size=(auction.num_vms, auction.num_resources))
    
    # Run first-price auction
    fp_allocation = first_price_auction(auction.vm_requirements, auction.resource_capacity, random_bids)
    
    # Print comparison
    print("\nFirst-Price Auction Allocation:")
    fp_df = pd.DataFrame(fp_allocation, 
                        index=[f'VM {i+1}' for i in range(auction.num_vms)],
                        columns=[f'Resource {i+1}' for i in range(auction.num_resources)])
    print(fp_df.round(2))
    
    # Calculate efficiency
    fp_total_allocated = np.sum(fp_allocation, axis=0)
    fp_efficiency = fp_total_allocated / auction.resource_capacity
    print("\nFirst-Price Resource Utilization Efficiency:")
    fp_eff_df = pd.DataFrame({
        'Resource': [f'Resource {i+1}' for i in range(auction.num_resources)],
        'Total Capacity': auction.resource_capacity,
        'Total Allocated': fp_total_allocated,
        'Efficiency (%)': fp_efficiency * 100
    })
    print(fp_eff_df.round(2))
    
    # Compare with Nash equilibrium
    ne_efficiency = np.sum(auction.allocations, axis=0) / auction.resource_capacity
    print("\nEfficiency Comparison (Nash Equilibrium vs First-Price):")
    comp_df = pd.DataFrame({
        'Resource': [f'Resource {i+1}' for i in range(auction.num_resources)],
        'Nash Equilibrium Efficiency (%)': ne_efficiency * 100,
        'First-Price Efficiency (%)': fp_efficiency * 100,
        'Difference (%)': (ne_efficiency - fp_efficiency) * 100
    })
    print(comp_df.round(2))