import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation


class BeliefVisualizer:
    def __init__(self, colormap):
        self.colormap = colormap
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup colormap display
        self.ax1.imshow(colormap, cmap='gray', vmin=0, vmax=1)
        self.ax1.set_title('Environment Map')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        
        # Setup belief display
        self.belief_img = self.ax2.imshow(np.zeros_like(colormap, dtype=float), 
                                          cmap='hot', vmin=0, vmax=1)
        self.ax2.set_title('Belief Distribution')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        plt.colorbar(self.belief_img, ax=self.ax2)
        
    def update_belief(self, belief):
        self.belief_img.set_data(belief)
        self.belief_img.set_clim(vmin=0, vmax=belief.max())
        self.fig.canvas.draw()
        
    def show(self):
        plt.show()
        
    def save_frame(self, filename):
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')


def plot_belief_evolution(beliefs, colormap, actions=None, observations=None):
    """
    Plot the evolution of belief over multiple timesteps
    
    Args:
        beliefs: List of belief arrays
        colormap: The environment map
        actions: List of actions taken (optional)
        observations: List of observations received (optional)
    """
    n_steps = len(beliefs)
    fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, belief in enumerate(beliefs):
        ax = axes[i]
        im = ax.imshow(belief, cmap='hot', vmin=0, vmax=np.max(beliefs))
        ax.set_title(f'Step {i}')
        
        if actions and i < len(actions):
            action_str = f"Action: {actions[i]}"
            ax.text(0.5, -0.1, action_str, ha='center', 
                   transform=ax.transAxes, fontsize=8)
        
        if observations and i < len(observations):
            obs_str = f"Obs: {observations[i]}"
            ax.text(0.5, -0.15, obs_str, ha='center', 
                   transform=ax.transAxes, fontsize=8)
    
    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def create_animation(beliefs, colormap, interval=500):
    """
    Create an animation of belief evolution
    
    Args:
        beliefs: List of belief arrays
        colormap: The environment map
        interval: Time between frames in milliseconds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Static colormap
    ax1.imshow(colormap, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Environment Map')
    
    # Animated belief
    belief_img = ax2.imshow(beliefs[0], cmap='hot', vmin=0, vmax=np.max(beliefs))
    ax2.set_title('Belief Distribution - Step 0')
    plt.colorbar(belief_img, ax=ax2)
    
    def animate(frame):
        belief_img.set_data(beliefs[frame])
        ax2.set_title(f'Belief Distribution - Step {frame}')
        return [belief_img]
    
    anim = FuncAnimation(fig, animate, frames=len(beliefs), 
                        interval=interval, blit=True)
    
    return anim