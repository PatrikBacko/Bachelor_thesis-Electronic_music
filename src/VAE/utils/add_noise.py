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


def add_noise(spectrogram, noise) -> callable:
    """
    Add noise to the spectrogram

    params:
        - spectrogram: spectrogram to add noise to
        - noise: noise array
    
    returns:
        - spectrogram with added noise
    """
    return spectrogram + noise.astype(np.float32)

def multiply_noise(spectrogram, noise) -> callable:
    """
    Multiply the spectrogram with noise

    params:
        - spectrogram: spectrogram to multiply with noise
        - noise: noise array
    
    returns:
        - spectrogram multiplied with noise
    """
    return spectrogram * noise.astype(np.float32)



def generate_noise(mean, variance, distribution, scope, operation) -> callable:
    """
    Generate noise function based on the given arguments, returns the function that applies the noise to the spectrogram

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
        scope_func = lambda spectrogram: dist_func(spectrogram.shape)
    elif scope == "column":
        scope_func = lambda spectrogram: np.outer(dist_func(spectrogram.shape[1]), np.ones(spectrogram.shape[0]))
    elif scope == "row":
        scope_func = lambda spectrogram: np.outer(np.ones(spectrogram.shape[1]), dist_func(spectrogram.shape[0]))
    elif scope == "entire_picture":
        scope_func = lambda spectrogram: np.full(spectrogram.shape, dist_func((1,))[0])

    #noise operation type
    if operation == "additive":
        noise_function = lambda spectrogram: add_noise(spectrogram, scope_func(spectrogram))
    elif operation == "multiplicative":
        noise_function = lambda spectrogram: multiply_noise(spectrogram, scope_func(spectrogram))

    return noise_function