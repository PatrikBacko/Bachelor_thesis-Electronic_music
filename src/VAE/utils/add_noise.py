import numpy as np

NOISE_GENERATING_DISTS = ["normal", "uniform", "constant"]
NOISE_OPERATION_TYPES = ["additive", "multiplicative"]
NOISE_SCOPE = ["pixel", "column", "row", "entire_picture"]

def normal_noise(mean: float, variance: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Generate normal noise with given mean and variance

    params:
        - mean: mean of the normal distribution
        - variance: variance of the normal distribution
        - shape: shape of the noise array
    
    returns:
        - noise array
    """
    return np.random.normal(mean, variance, shape)

def uniform_noise(mean: float, variance: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Generate uniform noise with given mean and variance

    params:
        - mean: mean of the uniform distribution
        - variance: variance of the uniform distribution
        - shape: shape of the noise array

    returns:
        - noise array
    """
    return np.random.uniform(mean-variance, mean+variance, shape)

def constant_noise(mean: float, variance: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Generate constant noise with given mean and variance

    params:
        - mean: mean of the constant distribution
        - variance: variance of the constant distribution
        - shape: shape of the noise array 

    returns:
        - noise array
    """
    return np.full(shape, mean)


def add_noise(spectogram, noise) -> callable:
    """
    Add noise to the spectogram

    params:
        - spectogram: spectogram to add noise to
        - noise: noise array
    
    returns:
        - spectogram with added noise
    """
    return spectogram + noise.astype(np.float32)

def multiply_noise(spectogram, noise) -> callable:
    """
    Multiply the spectogram with noise

    params:
        - spectogram: spectogram to multiply with noise
        - noise: noise array
    
    returns:
        - spectogram multiplied with noise
    """
    return spectogram * noise.astype(np.float32)



def generate_noise(mean, variance, distribution, scope, operation) -> callable:
    """
    Generate noise function based on the given arguments, returns the function that applies the noise to the spectogram

    params:
        - mean: mean of the noise distribution
        - variance: variance of the noise distribution
        - distribution: noise generating distribution
        - scope: noise scope
        - operation: noise operation type

    returns:
        - noise function
    """

    #noise generating distribution
    if distribution == "normal":
        dist_func = lambda shape: normal_noise(mean, variance, shape)
    elif distribution == "uniform":
        dist_func = lambda shape: uniform_noise(mean, variance, shape)
    elif distribution == "constant":
        dist_func = lambda shape: constant_noise(mean, variance, shape)

    #noise scope
    if scope == "pixel":
        scope_func = lambda spectogram: dist_func(spectogram.shape)
    elif scope == "column":
        scope_func = lambda spectogram: np.outer(dist_func(spectogram.shape[1]), np.ones(spectogram.shape[0]))
    elif scope == "row":
        scope_func = lambda spectogram: np.outer(np.ones(spectogram.shape[1]), dist_func(spectogram.shape[0]))
    elif scope == "entire_picture":
        scope_func = lambda spectogram: np.full(spectogram.shape, dist_func((1,))[0])

    #noise operation type
    if operation == "additive":
        noise_function = lambda spectogram: add_noise(spectogram, scope_func(spectogram))
    elif operation == "multiplicative":
        noise_function = lambda spectogram: multiply_noise(spectogram, scope_func(spectogram))

    return noise_function