"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import *
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
SAFETY_MARGIN = 0.4 #to be tuned 


# DONE: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # DONE: Define all relevant class members for your BO algorithm here.
        self.Kernel_f = Matern(length_scale=1, nu=2.5)
        self.Kernel_v = DotProduct(sigma_0=1.0) + Matern(length_scale=0.5, nu=2.5)

        self.sigma_f = 0.15
        self.sigma_v = 0.0001

        # initialize prior mean of f and v
        self.mu_f = 0
        self.mu_v = 4

        # initialize model 
        self.sampled_points = np.empty((0,1))
        self.obj_values = np.empty((0,1))
        self.SA_values = np.empty((0,1))

        # failure probability
        self.delta = 0.01

        # estimates for Lipschitz constants for v
        self.L_v = 7      # conservative estimate
        # self.L_v = 0.5**(1/4)/0.1   # length constant at denominator 

        # grid on which to evaluate functions 
        self.num_samples = 1001
        self.grid = np.linspace(*DOMAIN[0], self.num_samples)
        
        # initialized confidence bounds 
        self.l_f = np.full(self.num_samples, -np.inf)
        self.u_f = np.full(self.num_samples, np.inf)
        self.l_v = np.full(self.num_samples, -np.inf)
        self.u_v = np.full(self.num_samples, np.inf)

        # initialize set of grid indices of safe points 
        self.S = np.array([], dtype=int)
        # initialize set of grid indices of potential expander points
        self.G = np.array([], dtype=int)

        print("\n")

        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # DONE: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        # find formula for beta_t

        # compute beta_t schedule for f and v
        t = len(self.sampled_points) + 1 
        # estimate that can be tuned
        self.beta_f = 2 * np.log(2 * self.num_samples * (t**2) * (np.pi**2) / (6 * 0.1))        
        sqrt_beta_f = np.sqrt(self.beta_f)

        self.beta_v = 2 * np.log(2 * self.num_samples * (t**2) * (np.pi**2) / (6 * self.delta))        
        sqrt_beta_v = np.sqrt(self.beta_v)

        ##### compute upper and lower bounds for f and v
        # the conpute maximum lower bound for f
        
        # initialize arrays
        lower_bounds_f = np.zeros(self.num_samples)
        upper_bounds_f = np.zeros(self.num_samples)
        lower_bounds_v = np.zeros(self.num_samples)
        upper_bounds_v = np.zeros(self.num_samples)

        # vectorized bound computation
        mu_f, sigma_f, mu_v, sigma_v = self._fast_gp_predict(self.grid)

        lower_bounds_f = mu_f - sqrt_beta_f * sigma_f
        upper_bounds_f = mu_f + sqrt_beta_f * sigma_f
        lower_bounds_v = mu_v - sqrt_beta_v * sigma_v
        upper_bounds_v = mu_v + sqrt_beta_v * sigma_v   


        # construction of C_t sets for f and v
        self.l_f = np.maximum(self.l_f, lower_bounds_f)
        self.u_f = np.minimum(self.u_f, upper_bounds_f)
        self.l_v = np.maximum(self.l_v, lower_bounds_v)
        self.u_v = np.minimum(self.u_v, upper_bounds_v)


        # construction of S_t set for v 
        new_safe_indices = set(self.S)  
        
        for idx in self.S:
            x_s = self.grid[idx]  
            safe_mask = (self.u_v[idx] + self.L_v * np.abs(x_s - self.grid)) < SAFETY_THRESHOLD - SAFETY_MARGIN
            new_safe_indices.update(np.where(safe_mask)[0])
        
        self.S = np.array(sorted(new_safe_indices), dtype=int)


        # construction of G_t for v
        expanders_counts = []
        all_indices = np.arange(len(self.grid))  
        not_in_S = np.setdiff1d(all_indices, self.S)

        for idx_s in self.S:
            x_s = self.grid[idx_s]  
            l_v_not_in_S = self.l_v[not_in_S]
            expanders_mask = (l_v_not_in_S + self.L_v * np.abs(x_s - self.grid[not_in_S])) < SAFETY_THRESHOLD - SAFETY_MARGIN
            count = np.count_nonzero(expanders_mask)
            expanders_counts.append(count)

        expanders_counts = np.array(expanders_counts)
        self.G = self.S[expanders_counts > 0]

        self.max_lower_f = max(lower_bounds_f)

        # 1. Identify M_t (Potential Maximizers)
        # Definition: Safe points where u_f >= max_l_f of the safe set
        # (Note: logic assumes self.S is already updated correctly)
        safe_l_f = self.l_f[self.S]
        self.max_lower_f = np.max(safe_l_f) if len(safe_l_f) > 0 else -np.inf

        # Vectorized check for M_t
        is_in_M = (self.u_f >= self.max_lower_f)
        # Must also be safe to be in M
        M_indices = np.array([i for i in self.S if is_in_M[i]])

        # 2. Combine G_t and M_t to get all valid target indices
        # (self.G is already computed in your code)
        valid_set = np.union1d(self.G, M_indices).astype(int)
        if len(valid_set) == 0:
            valid_set = self.S  # fallback to full safe set

        # 3. Store the coordinates of these valid points for the acquisition function
        self.valid_coords = self.grid[valid_set]
        

        x_bfgs = self.optimize_acquisition_function()
        # span the optimize point to the grid
        
        dists = np.abs(self.grid[self.S] - x_bfgs)
        best_local_idx = np.argmin(dists)
        best_grid_idx = self.S[best_local_idx]
        
        return np.array([[self.grid[best_grid_idx]]])

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # DONE: Implement the acquisition function you want to optimize.
        # update the gaussian processes
        x_val = x.item()

        l_f_x = np.interp(x_val, self.grid, self.l_f)
        u_f_x = np.interp(x_val, self.grid, self.u_f)
        w_f_x = u_f_x

        dist_to_valid = np.min(np.abs(self.valid_coords - x_val))
        
        # Define a tolerance (e.g., half a grid step)
        # If we are essentially "on" a valid point, penalty is 0.
        grid_step = self.grid[1] - self.grid[0]
        
        if dist_to_valid <= grid_step * 0.5:
            # Inside the valid region return
            return float(w_f_x)
        else:
            # Outside valid region: Return a "slope" guiding back to safety.
            # The value should be negative to ensure it's worse than any valid point.
            # Slope factor (e.g., 10.0) makes it steep.
            return float(-10.0 * dist_to_valid)
        
        raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # DONE: Add the observed data {x, f, v} to your model.
        x = float(np.atleast_1d(x).ravel()[0])

        # add points to the model
        self.sampled_points = np.vstack([self.sampled_points, [[x]]])
        self.obj_values = np.vstack([self.obj_values, [f]])
        self.SA_values = np.vstack([self.SA_values, [v]]) 

        # Compute the cached matrices for fitting the GPs
        n = self.sampled_points.shape[0]
        I = np.eye(n)
        KK_f = self.Kernel_f(self.sampled_points, self.sampled_points)
        self.center_mat_f = np.linalg.inv(KK_f + self.sigma_f**2 * I)

        KK_v = self.Kernel_v(self.sampled_points, self.sampled_points)
        self.center_mat_v = np.linalg.inv(KK_v + self.sigma_v**2 * I)

        # add first point to the safe set (find closest grid index)
        if len(self.sampled_points) == 1:
            distances = np.abs(self.grid - x)
            closest_index = np.argmin(distances)
            self.S = np.array([closest_index], dtype=int)

        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # DONE: Return your predicted safe optimum of f.

        safe_objs = np.where(self.SA_values < SAFETY_THRESHOLD, self.obj_values, -np.inf)
        safe_optimum_index = np.argmax(safe_objs)
        return float(self.sampled_points[safe_optimum_index, 0])  
        raise NotImplementedError

    def _fast_gp_predict(self, x):
        """Fast GP prediction using cached matrices."""
        # Check dimentions
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n = self.sampled_points.shape[0]
        n_samples = x.shape[0]
        
        if n == 0:
            return np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 4.0), np.ones(n_samples)  # Prior values
        
        # Predict f
        k_f_x = self.Kernel_f(self.sampled_points, x)
        mu_f = (k_f_x.T @ self.center_mat_f @ self.obj_values).flatten()
        var_f_prior = self.Kernel_f.diag(x)
        v_vec_f = self.center_mat_f @ k_f_x
        var_f_reduction = np.sum(k_f_x * v_vec_f, axis=0)
        sigma_f = np.sqrt(np.maximum(0, var_f_prior - var_f_reduction))
        
        # Predict v
        k_v_x = self.Kernel_v(self.sampled_points, x)
        mu_v = (k_v_x.T @ self.center_mat_v @ (self.SA_values - 4)).flatten() + 4
        var_v_prior = self.Kernel_v.diag(x)
        v_vec_v = self.center_mat_v @ k_v_x
        var_v_reduction = np.sum(k_v_x * v_vec_v, axis=0)
        sigma_v = np.sqrt(np.maximum(0, var_v_prior - var_v_reduction))
        
        return mu_f, sigma_f, mu_v, sigma_v
    
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
