import numpy as np

class HMM():
    def __init__(self, Observations, Transition, Emission, Initial_distribution):

        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        alpha = np.zeros((len(self.Observations), len(self.Initial_distribution)))

        for i in range(len(self.Initial_distribution)):
            alpha[0, i] = self.Initial_distribution[i] * self.Emission[i, self.Observations[0]]

        for t in range(1, len(self.Observations)):
            for i in range(len(self.Initial_distribution)):
                alpha[t, i] = np.dot(alpha[t-1], self.Transition[:, i]) * self.Emission[i, self.Observations[t]]

        return alpha

    def backward(self):
        beta = np.zeros((len(self.Observations), len(self.Initial_distribution)))
        beta[len(self.Observations)-1, :] = 1.0

        for t in range(len(self.Observations)-2, -1, -1):
            for i in range(len(self.Initial_distribution)):
                beta[t, i] = np.sum(self.Transition[i, :] * self.Emission[:, self.Observations[t+1]] * beta[t+1, :])

        return beta

    def gamma_comp(self, alpha, beta):
        P_obs = np.sum(alpha[-1, :])
        gamma = (alpha * beta) / P_obs
        
        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.zeros((alpha.shape[0] - 1, alpha.shape[1], alpha.shape[1]))
        
        for t in range(alpha.shape[0] - 1):
            denominator = 0
            for i in range(alpha.shape[1]):
                for j in range(alpha.shape[1]):
                    denominator += alpha[t, i] * self.Transition[i, j] *self.Emission[j, self.Observations[t+1]] * beta[t+1, j]
            for i in range(alpha.shape[1]):
                for j in range(alpha.shape[1]):
                    numerator = alpha[t, i] * self.Transition[i, j] *self.Emission[j, self.Observations[t+1]] * beta[t+1, j]
                    xi[t, i, j] = numerator / denominator

        return xi

    def update(self, alpha, beta, gamma, xi):
        new_init_state = gamma[0, :]

        T_prime = np.zeros((len(self.Initial_distribution), len(self.Initial_distribution)))
        for i in range(len(self.Initial_distribution)):
            denominator = np.sum(gamma[:-1, i])
            for j in range(len(self.Initial_distribution)):
                numerator = np.sum(xi[:, i, j])
                T_prime[i, j] = numerator / denominator

        M_prime = np.zeros((len(self.Initial_distribution), self.Emission.shape[1]))
        for i in range(len(self.Initial_distribution)):
            denominator = np.sum(gamma[:, i])
            for k in range(self.Emission.shape[1]):
                numerator = 0
                for t in range(len(self.Observations)):
                    if self.Observations[t] == k:
                        numerator += gamma[t, i]
                M_prime[i, k] = numerator / denominator

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.sum(alpha[-1, :])  

        Transition = self.Transition
        Emission = self.Emission
        pi = self.Initial_distribution

        self.Transition = T_prime
        self.Emission = M_prime
        self.Initial_distribution = new_init_state

        alpha_prime = self.forward()
        P_prime = np.sum(alpha_prime[-1, :])

        # Restore old parameters
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = pi

        return P_original, P_prime
