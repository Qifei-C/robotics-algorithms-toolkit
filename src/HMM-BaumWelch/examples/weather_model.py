import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hmm import HMM
import matplotlib.pyplot as plt


def create_weather_hmm():
    """
    Create an HMM for weather prediction based on activities
    
    Hidden states: 0 = Rainy, 1 = Sunny
    Observations: 0 = Stay Inside, 1 = Go for Walk, 2 = Go Shopping
    """
    
    # True parameters (what we're trying to learn)
    true_transition = np.array([
        [0.7, 0.3],  # Rainy -> [Rainy, Sunny]
        [0.2, 0.8]   # Sunny -> [Rainy, Sunny]
    ])
    
    true_emission = np.array([
        [0.6, 0.2, 0.2],  # Rainy: [Inside, Walk, Shop]
        [0.1, 0.7, 0.2]   # Sunny: [Inside, Walk, Shop]
    ])
    
    true_initial = np.array([0.5, 0.5])
    
    # Generate synthetic observations
    np.random.seed(42)
    observations = generate_observations(true_transition, true_emission, 
                                       true_initial, length=100)
    
    print("Generated observation sequence (first 20):")
    activity_names = ['Stay Inside', 'Go for Walk', 'Go Shopping']
    print([activity_names[obs] for obs in observations[:20]])
    print()
    
    # Initialize with random parameters
    n_states = 2
    n_obs_types = 3
    
    init_transition = np.random.dirichlet(np.ones(n_states), size=n_states)
    init_emission = np.random.dirichlet(np.ones(n_obs_types), size=n_states)
    init_initial = np.random.dirichlet(np.ones(n_states))
    
    print("Initial random parameters:")
    print("Transition matrix:")
    print(init_transition)
    print("\nEmission matrix:")
    print(init_emission)
    print("\nInitial distribution:")
    print(init_initial)
    print()
    
    # Create HMM and train
    hmm = HMM(observations, init_transition, init_emission, init_initial)
    
    # Run Baum-Welch for multiple iterations
    n_iterations = 50
    log_likelihoods = []
    
    for i in range(n_iterations):
        # Forward-backward
        alpha = hmm.forward()
        beta = hmm.backward()
        gamma = hmm.gamma_comp(alpha, beta)
        xi = hmm.xi_comp(alpha, beta, gamma)
        
        # Get log likelihood
        log_likelihood = np.log(np.sum(alpha[-1, :]))
        log_likelihoods.append(log_likelihood)
        
        # Update parameters
        new_trans, new_emit, new_init = hmm.update(alpha, beta, gamma, xi)
        
        # Update HMM parameters
        hmm.Transition = new_trans
        hmm.Emission = new_emit
        hmm.Initial_distribution = new_init
        
        if i % 10 == 0:
            print(f"Iteration {i}: Log-likelihood = {log_likelihood:.4f}")
    
    print(f"\nFinal log-likelihood: {log_likelihoods[-1]:.4f}")
    
    # Compare learned parameters with true parameters
    print("\n" + "="*50)
    print("PARAMETER COMPARISON")
    print("="*50)
    
    print("\nTransition Matrix:")
    print("True:")
    print(true_transition)
    print("Learned:")
    print(hmm.Transition)
    print("Difference:")
    print(np.abs(true_transition - hmm.Transition))
    
    print("\nEmission Matrix:")
    print("True:")
    print(true_emission)
    print("Learned:")
    print(hmm.Emission)
    print("Difference:")
    print(np.abs(true_emission - hmm.Emission))
    
    print("\nInitial Distribution:")
    print("True:", true_initial)
    print("Learned:", hmm.Initial_distribution)
    print("Difference:", np.abs(true_initial - hmm.Initial_distribution))
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(log_likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('Baum-Welch Convergence')
    plt.grid(True)
    plt.savefig('baum_welch_convergence.png')
    print("\nSaved convergence plot to baum_welch_convergence.png")
    
    # Predict most likely weather sequence
    gamma_final = hmm.gamma_comp(alpha, beta)
    predicted_states = np.argmax(gamma_final, axis=1)
    
    print("\nPredicted weather for first 20 observations:")
    weather_names = ['Rainy', 'Sunny']
    print("Observations:", [activity_names[obs] for obs in observations[:20]])
    print("Predictions: ", [weather_names[state] for state in predicted_states[:20]])
    
    return hmm


def generate_observations(transition, emission, initial, length=100):
    """Generate observations from true HMM parameters"""
    n_states = transition.shape[0]
    observations = []
    
    # Start with initial distribution
    current_state = np.random.choice(n_states, p=initial)
    
    for _ in range(length):
        # Generate observation from current state
        obs = np.random.choice(emission.shape[1], p=emission[current_state])
        observations.append(obs)
        
        # Transition to next state
        current_state = np.random.choice(n_states, p=transition[current_state])
    
    return observations


def weather_prediction_demo():
    """
    Demonstrate using trained HMM for weather prediction
    """
    print("\n" + "="*50)
    print("WEATHER PREDICTION DEMO")
    print("="*50)
    
    # Create and train HMM
    hmm = create_weather_hmm()
    
    # New observation sequence
    new_observations = [0, 0, 1, 1, 2, 1, 1]  # Activities for a week
    activity_names = ['Stay Inside', 'Go for Walk', 'Go Shopping']
    
    print("\nNew observation sequence:")
    print([activity_names[obs] for obs in new_observations])
    
    # Create new HMM with these observations
    hmm_predict = HMM(new_observations, hmm.Transition, 
                     hmm.Emission, hmm.Initial_distribution)
    
    # Run forward-backward
    alpha = hmm_predict.forward()
    beta = hmm_predict.backward()
    gamma = hmm_predict.gamma_comp(alpha, beta)
    
    # Get most likely states
    predicted_states = np.argmax(gamma, axis=1)
    weather_names = ['Rainy', 'Sunny']
    
    print("\nPredicted weather:")
    for i, (obs, state) in enumerate(zip(new_observations, predicted_states)):
        prob = gamma[i, state]
        print(f"Day {i+1}: {activity_names[obs]} -> "
              f"{weather_names[state]} (confidence: {prob:.2f})")


if __name__ == "__main__":
    weather_prediction_demo()