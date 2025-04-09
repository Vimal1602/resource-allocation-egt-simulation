import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
import random

class EvolutionarySecurityGame:
    """
    A simulation of security threat detection in distributed networks using evolutionary game theory.
    Models the interaction between attackers and defenders in a network environment and how security
    strategies evolve over time based on fitness and payoffs.
    """
    
    def __init__(self, num_nodes=10, time_horizon=100):
        self.num_nodes = num_nodes  # Number of network nodes
        self.time_horizon = time_horizon  # Simulation time
        
        # Strategy sets
        self.attacker_strategies = ["DoS Attack", "Data Theft", "Passive Reconnaissance"]
        self.defender_strategies = ["Firewall", "IDS", "Honeypot"]
        
        self.num_attacker_strats = len(self.attacker_strategies)
        self.num_defender_strats = len(self.defender_strategies)
        
        # Payoff matrices
        # Format: attacker_payoff[attacker_strategy][defender_strategy]
        self.attacker_payoff = np.array([
            [-5, -10, -20],     # DoS payoffs against different defenses
            [15, -15, -10],     # Data Theft payoffs
            [5, 0, -30]         # Passive Reconnaissance payoffs
        ])
        
        # Format: defender_payoff[attacker_strategy][defender_strategy]
        self.defender_payoff = np.array([
            [-15, -5, 0],      # Defender payoffs against DoS
            [-30, 10, 5],      # Defender payoffs against Data Theft
            [-5, -5, 20]       # Defender payoffs against Reconnaissance
        ])
        
        # Initial strategy distribution (population shares)
        self.attacker_population = np.ones(self.num_attacker_strats) / self.num_attacker_strats
        self.defender_population = np.ones(self.num_defender_strats) / self.num_defender_strats
        
        # Cost of implementing each strategy
        self.defender_costs = np.array([5, 10, 15])  # Costs for Firewall, IDS, Honeypot
        
        # Track history
        self.attacker_history = [self.attacker_population.copy()]
        self.defender_history = [self.defender_population.copy()]
        self.time_points = [0]
        
        # Attack success rates
        self.attack_success_history = []
        
    def calculate_expected_payoffs(self):
        """
        Calculate the expected payoffs for each strategy based on 
        the current population distribution.
        """
        # Expected payoffs for attackers
        attacker_expected = np.zeros(self.num_attacker_strats)
        for a in range(self.num_attacker_strats):
            for d in range(self.num_defender_strats):
                attacker_expected[a] += self.attacker_payoff[a, d] * self.defender_population[d]
        
        # Expected payoffs for defenders
        defender_expected = np.zeros(self.num_defender_strats)
        for d in range(self.num_defender_strats):
            for a in range(self.num_attacker_strats):
                defender_expected[d] += self.defender_payoff[a, d] * self.attacker_population[a]
            # Subtract the cost of implementing the defense strategy
            defender_expected[d] -= self.defender_costs[d]
            
        return attacker_expected, defender_expected
    
    def replicator_dynamics(self, t, y):
        """
        Implement the replicator dynamics differential equations.
        This determines how strategy populations evolve over time.
        """
        # Split the combined state vector into attacker and defender populations
        attacker_pop = y[:self.num_attacker_strats]
        defender_pop = y[self.num_attacker_strats:]
        
        # Store the current state
        self.attacker_population = attacker_pop
        self.defender_population = defender_pop
        
        # Calculate expected payoffs
        attacker_payoffs, defender_payoffs = self.calculate_expected_payoffs()
        
        # Calculate average payoffs
        avg_attacker_payoff = np.sum(attacker_pop * attacker_payoffs)
        avg_defender_payoff = np.sum(defender_pop * defender_payoffs)
        
        # Replicator dynamics equations
        d_attacker = attacker_pop * (attacker_payoffs - avg_attacker_payoff)
        d_defender = defender_pop * (defender_payoffs - avg_defender_payoff)
        
        # Combine into a single derivative vector
        derivatives = np.concatenate([d_attacker, d_defender])
        
        return derivatives
    
    def simulate(self):
        """
        Run the evolutionary game simulation over the specified time horizon.
        """
        # Initialize with combined state vector of attacker and defender populations
        y0 = np.concatenate([self.attacker_population, self.defender_population])
        
        # Time points to solve for
        t_span = (0, self.time_horizon)
        t_eval = np.linspace(0, self.time_horizon, 100)
        
        # Solve the differential equations
        result = solve_ivp(
            self.replicator_dynamics, 
            t_span, 
            y0, 
            method='RK45', 
            t_eval=t_eval
        )
        
        # Store results
        for i, t in enumerate(result.t):
            attacker_pop = result.y[:self.num_attacker_strats, i]
            defender_pop = result.y[self.num_attacker_strats:, i]
            
            self.attacker_history.append(attacker_pop)
            self.defender_history.append(defender_pop)
            self.time_points.append(t)
            
            # Calculate the attack success rate at this time point
            attack_success = self.calculate_attack_success(attacker_pop, defender_pop)
            self.attack_success_history.append(attack_success)
        
        return result
    
    def calculate_attack_success(self, attacker_pop, defender_pop):
        """
        Calculate the overall success rate of attacks based on current strategy distributions.
        """
        success_rate = 0
        for a in range(self.num_attacker_strats):
            for d in range(self.num_defender_strats):
                # Positive attacker payoff indicates successful attack
                if self.attacker_payoff[a, d] > 0:
                    success_probability = attacker_pop[a] * defender_pop[d]
                    success_rate += success_probability * self.attacker_payoff[a, d] / 20.0  # Normalize
        
        return success_rate
    
    def plot_population_dynamics(self):
        """
        Plot the evolution of strategy populations over time.
        """
        attacker_history = np.array(self.attacker_history)
        defender_history = np.array(self.defender_history)
        time_points = np.array(self.time_points)
        
        plt.figure(figsize=(15, 10))
        
        # Plot attacker strategies
        plt.subplot(2, 1, 1)
        for i, strategy in enumerate(self.attacker_strategies):
            plt.plot(time_points, attacker_history[:, i], label=strategy)
        
        plt.title('Evolution of Attacker Strategies')
        plt.xlabel('Time')
        plt.ylabel('Population Share')
        plt.legend()
        plt.grid(True)
        
        # Plot defender strategies
        plt.subplot(2, 1, 2)
        for i, strategy in enumerate(self.defender_strategies):
            plt.plot(time_points, defender_history[:, i], label=strategy)
        
        plt.title('Evolution of Defender Strategies')
        plt.xlabel('Time')
        plt.ylabel('Population Share')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_attack_success_rate(self):
        """
        Plot the evolution of attack success rate over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points[1:], self.attack_success_history)
        plt.title('Evolution of Attack Success Rate')
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.show()
    
    def print_results(self):
        """
        Print detailed results of the evolutionary game.
        """
        print("\n===== EVOLUTIONARY SECURITY GAME RESULTS =====")
        
        # Print initial and final strategy distributions
        print("\nInitial Strategy Distribution:")
        print("Attackers:")
        for i, strat in enumerate(self.attacker_strategies):
            print(f"  {strat}: {self.attacker_history[0][i]:.3f}")
        print("Defenders:")
        for i, strat in enumerate(self.defender_strategies):
            print(f"  {strat}: {self.defender_history[0][i]:.3f}")
        
        print("\nFinal Strategy Distribution:")
        print("Attackers:")
        for i, strat in enumerate(self.attacker_strategies):
            print(f"  {strat}: {self.attacker_history[-1][i]:.3f}")
        print("Defenders:")
        for i, strat in enumerate(self.defender_strategies):
            print(f"  {strat}: {self.defender_history[-1][i]:.3f}")
        
        # Calculate and print the dominant strategies
        dominant_attacker = np.argmax(self.attacker_history[-1])
        dominant_defender = np.argmax(self.defender_history[-1])
        
        print(f"\nDominant Attacker Strategy: {self.attacker_strategies[dominant_attacker]}")
        print(f"Dominant Defender Strategy: {self.defender_strategies[dominant_defender]}")
        
        # Print final attack success rate
        print(f"\nFinal Attack Success Rate: {self.attack_success_history[-1]:.4f}")
        
        # Print payoff matrix for reference
        print("\nAttacker Payoff Matrix:")
        att_payoff_df = pd.DataFrame(
            self.attacker_payoff, 
            index=self.attacker_strategies,
            columns=self.defender_strategies
        )
        print(att_payoff_df)
        
        print("\nDefender Payoff Matrix:")
        def_payoff_df = pd.DataFrame(
            self.defender_payoff, 
            index=self.attacker_strategies,
            columns=self.defender_strategies
        )
        print(def_payoff_df)
    
    def visualize_payoff_matrices(self):
        """
        Create heatmaps to visualize the payoff matrices.
        """
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(
            self.attacker_payoff, 
            annot=True, 
            fmt=".0f", 
            cmap="RdBu_r",
            xticklabels=self.defender_strategies,
            yticklabels=self.attacker_strategies
        )
        plt.title('Attacker Payoff Matrix')
        plt.xlabel('Defender Strategy')
        plt.ylabel('Attacker Strategy')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(
            self.defender_payoff, 
            annot=True, 
            fmt=".0f", 
            cmap="RdBu_r",
            xticklabels=self.defender_strategies,
            yticklabels=self.attacker_strategies
        )
        plt.title('Defender Payoff Matrix')
        plt.xlabel('Defender Strategy')
        plt.ylabel('Attacker Strategy')
        
        plt.tight_layout()
        plt.show()
    
    def run_network_simulation(self, num_time_steps=50):
        """
        Run a practical network simulation to demonstrate the impact of evolved strategies.
        """
        print("\n===== NETWORK SIMULATION =====")
        
        # Get final evolved strategy distributions
        attacker_strat_dist = self.attacker_history[-1]
        defender_strat_dist = self.defender_history[-1]
        
        # Initialize network nodes with random defender strategies
        node_strategies = np.random.choice(
            range(self.num_defender_strats),
            size=self.num_nodes,
            p=defender_strat_dist
        )
        
        # Track successful and unsuccessful attacks
        successful_attacks = 0
        total_attacks = 0
        
        # Record attack history for visualization
        attack_history = []
        
        print(f"Simulating network with {self.num_nodes} nodes over {num_time_steps} time steps...")
        
        # Run simulation
        for t in range(num_time_steps):
            # Each time step, attackers target a random node
            for _ in range(3):  # Assume 3 attack attempts per time step
                # Select random node to attack
                target_node = random.randint(0, self.num_nodes - 1)
                defender_strategy = node_strategies[target_node]
                
                # Choose attacker strategy based on evolved distribution
                attacker_strategy = np.random.choice(
                    range(self.num_attacker_strats),
                    p=attacker_strat_dist
                )
                
                # Determine if attack is successful
                attack_payoff = self.attacker_payoff[attacker_strategy, defender_strategy]
                
                # Positive payoff indicates successful attack
                is_successful = attack_payoff > 0
                
                if is_successful:
                    successful_attacks += 1
                total_attacks += 1
                
                # Record attack details
                attack_history.append({
                    'time': t,
                    'target': target_node,
                    'attacker_strategy': self.attacker_strategies[attacker_strategy],
                    'defender_strategy': self.defender_strategies[defender_strategy],
                    'success': is_successful,
                    'payoff': attack_payoff
                })
            
            # Occasionally adapt node strategies based on attack experiences
            if t % 10 == 0 and t > 0:
                # Update a random subset of nodes based on observed performance
                nodes_to_update = random.sample(range(self.num_nodes), self.num_nodes // 3)
                
                for node in nodes_to_update:
                    # Choose a potentially better strategy based on evolved distribution
                    node_strategies[node] = np.random.choice(
                        range(self.num_defender_strats),
                        p=defender_strat_dist
                    )
        
        # Calculate final success rate
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
        print(f"Attack Success Rate: {success_rate:.2%} ({successful_attacks} successful out of {total_attacks} attacks)")
        
        # Analyze strategy effectiveness
        attack_df = pd.DataFrame(attack_history)
        
        if not attack_df.empty:
            # Strategy success rates
            print("\nAttacker Strategy Success Rates:")
            attacker_success = attack_df.groupby('attacker_strategy')['success'].agg(['mean', 'count'])
            print(attacker_success)
            
            print("\nDefender Strategy Success Rates:")
            defender_success = attack_df.groupby('defender_strategy')['success'].agg(['mean', 'count'])
            defender_success['mean'] = 1 - defender_success['mean']  # Convert to defense success rate
            print(defender_success)
            
            # Plot attack success over time
            plt.figure(figsize=(10, 6))
            attack_over_time = attack_df.groupby('time')['success'].mean()
            plt.plot(attack_over_time.index, attack_over_time.values)
            plt.title('Attack Success Rate Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Success Rate')
            plt.grid(True)
            plt.show()
        
        return attack_history

# Running the simulation
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    print("===== Security Threat Detection using Evolutionary Game Theory =====")
    print("Simulating attacker-defender dynamics in distributed networks...")
    
    # Initialize the game
    game = EvolutionarySecurityGame(num_nodes=20, time_horizon=100)
    
    # Run the evolutionary simulation
    print("\nRunning evolutionary simulation...")
    result = game.simulate()
    
    # Print and visualize results
    game.print_results()
    game.visualize_payoff_matrices()
    game.plot_population_dynamics()
    game.plot_attack_success_rate()
    
    # Run practical network simulation
    attack_history = game.run_network_simulation(num_time_steps=100)
    

    print("\nSimulation complete! The results demonstrate how security strategies evolve")
    print("over time in response to changing attack patterns, and how this affects")
    print("the overall security posture of the distributed network.")

    print("\nSimulation complete! The results demonstrate how security strategies evolve")
    print("over time in response to changing attack patterns, and how this affects")
    print("the overall security posture of the distributed network.")
    
    # Perform additional analysis on equilibrium strategies
    print("\n===== EQUILIBRIUM ANALYSIS =====")
    
    # Get final strategy distributions
    final_attacker_dist = game.attacker_history[-1]
    final_defender_dist = game.defender_history[-1]