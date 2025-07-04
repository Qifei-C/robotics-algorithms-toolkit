import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.histogram_filter import HistogramFilter
from src.visualizer import BeliefVisualizer, plot_belief_evolution


def simple_demo():
    """
    Simple demonstration of the histogram filter
    """
    # Create a simple 4x4 grid world
    colormap = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    
    # Initialize filter
    filter = HistogramFilter()
    
    # Start with uniform belief
    belief = np.ones_like(colormap, dtype=float) / colormap.size
    
    # Define a sequence of actions and observations
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Move in a square
    observations = [1, 0, 1, 0]  # What the robot observes
    
    beliefs = [belief]
    
    print("Initial belief (uniform):")
    print(belief)
    print()
    
    # Run filter
    for i, (action, obs) in enumerate(zip(actions, observations)):
        belief = filter.histogram_filter(colormap, belief, action, obs)
        beliefs.append(belief)
        
        print(f"Step {i+1}: Action={action}, Observation={obs}")
        print(f"Max belief at position: {np.unravel_index(np.argmax(belief), belief.shape)}")
        print(f"Max belief value: {np.max(belief):.3f}")
        print()
    
    # Visualize results
    fig = plot_belief_evolution(beliefs, colormap, actions, observations)
    fig.savefig('belief_evolution.png')
    print("Saved belief evolution to belief_evolution.png")


def interactive_demo():
    """
    Interactive demonstration where user can input actions
    """
    # Create a larger grid world
    colormap = np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1]
    ])
    
    filter = HistogramFilter()
    visualizer = BeliefVisualizer(colormap)
    
    # Start with uniform belief
    belief = np.ones_like(colormap, dtype=float) / colormap.size
    visualizer.update_belief(belief)
    
    print("Interactive Histogram Filter Demo")
    print("Actions: u=up, d=down, l=left, r=right, q=quit")
    print("The robot will observe the color at its new position")
    print()
    
    action_map = {
        'u': (0, -1),
        'd': (0, 1),
        'l': (-1, 0),
        'r': (1, 0)
    }
    
    while True:
        action_input = input("Enter action (u/d/l/r/q): ").lower()
        
        if action_input == 'q':
            break
            
        if action_input not in action_map:
            print("Invalid action. Use u/d/l/r/q")
            continue
            
        action = action_map[action_input]
        
        # Simulate observation (in reality, this would come from sensor)
        # For demo, we'll assume the robot observes the most likely position
        most_likely_pos = np.unravel_index(np.argmax(belief), belief.shape)
        observation = colormap[most_likely_pos]
        
        # Update belief
        belief = filter.histogram_filter(colormap, belief, action, observation)
        
        # Update visualization
        visualizer.update_belief(belief)
        
        print(f"Action: {action}, Observation: {observation}")
        print(f"Most likely position: {most_likely_pos}")
        print(f"Confidence: {np.max(belief):.3f}")
        print()
    
    visualizer.show()


def noise_analysis():
    """
    Analyze how different noise levels affect localization
    """
    colormap = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    
    # Test different sensor accuracy levels
    sensor_accuracies = [0.99, 0.9, 0.8, 0.7, 0.6]
    
    for accuracy in sensor_accuracies:
        # Create modified filter with different accuracy
        filter = HistogramFilter()
        
        # Modify the observation probabilities in the filter
        # (This would require modifying the original class to accept parameters)
        print(f"Sensor accuracy: {accuracy}")
        
        # Run a standard sequence
        belief = np.ones_like(colormap, dtype=float) / colormap.size
        actions = [(1, 0), (1, 0), (0, 1), (0, 1)]
        observations = [1, 0, 1, 0]
        
        for action, obs in zip(actions, observations):
            belief = filter.histogram_filter(colormap, belief, action, obs)
        
        max_belief = np.max(belief)
        print(f"Final maximum belief: {max_belief:.3f}")
        print()


if __name__ == "__main__":
    print("Running simple demo...")
    simple_demo()
    
    print("\n" + "="*50 + "\n")
    
    print("Running noise analysis...")
    noise_analysis()
    
    print("\n" + "="*50 + "\n")
    
    # Uncomment to run interactive demo
    # print("Starting interactive demo...")
    # interactive_demo()