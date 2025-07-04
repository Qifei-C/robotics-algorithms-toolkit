import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        nrows, ncols = cmap.shape
        new_belief = np.zeros_like(belief, dtype=float)
        dx, dy = action
        
        # corrected_cmap = np.flipud(cmap)  

        for r in range(nrows):
            for c in range(ncols):
                prob = belief[r, c] 
                if prob == 0:
                    continue  

                nr, nc = r - dy, c + dx
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    new_belief[nr, nc] += 0.9 * prob
                    new_belief[r, c] += 0.1 * prob
                else:
                    new_belief[r, c] += 0.9 * prob

        for r in range(nrows):
            for c in range(ncols):
                if cmap[r, c] == observation:
                    new_belief[r, c] *= 0.9  
                else:
                    new_belief[r, c] *= 0.1  

        norm_factor = np.sum(new_belief)
        if norm_factor > 0:
            new_belief /= norm_factor
            
        return new_belief
