from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

from probability import normal_cdf

# The normal cdf is the probability the variable is below a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float,
                             mu: float=0,
                             sigma: float=1) -> float:
    """The probability that N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    return 1 - normal_probability_between(lo, hi, mu, sigma)

from probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """Returns the z for which P(Z == z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """Returns the z for which P(Z == z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float=0,
                            sigma: float=1) -> Tuple[float, float]:
    """
    Returns the  symmetric (about the mean) bounds
    that contain teh specified probability
    """
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and simga based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our orginal interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability          # 0.887

hi = normal_upper_bound(0.5, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability          # 0.936

def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma)
    """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)
    
# If we were to see 530 heads we would compute
two_sided_p_value(529.5, mu_0, sigma_0)

#Can run this experiment to prove this to ourselves
import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0       # Count # of heads
                    for _ in range(1000))                   # in 1000 flips
    if num_heads >= 530 or num_heads <= 470:                # and count how often
        extreme_value_count += 1                            # the # is 'extreme'

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 68, f"{extreme_value_count}"

# if we instead saw 532 heads the p-value would be
two_sided_p_value(531.5, mu_0, sigma_0)         # 0.0463

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_p_value(524.5, mu_0, sigma_0) # 0.061
upper_p_value(526.5, mu_0, sigma_0) # 0.047

# Confidence Intervals

# Example we do 1000 flips of a coin and get 525 heads out of 1000 flips
# we can estimate that are p = 0.525, but how confident are we in that value
# CLT tells that the average of the bernoulli variables should be approx
# normal, with mean p and std = math.sqrt(p * (1 - p) / 1000)
# However we don't know p so we use our p_hat of 0.525

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)       # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)        # [0.4940, 0.5560]

# if we had instead seen 540 heads
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)

normal_two_sided_bounds(0.95, mu, sigma)        # [0.5091, 0.5709]

# With this p_hat value we see that p = 0.5 or are fair coin hypthesis doesn't
# lie within the confidence interval so we can conclude that the coin is biased

# P-Hacking
from typing import List

def run_experiment() -> List[bool]:
    """Flips a fair coin 1000 times, True = Heads, False = Tails"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """Using 5% significance levels"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                     for experiment in experiments
                     if reject_fairness(experiment)])

assert num_rejections == 46

# Example A/B testing

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

z = a_b_test_statistic(1000, 200, 1000, 180)        # -1.14
two_sided_p_value(z)                                # 0.254

# This is saying that their is about a 25% percent chance we would see this 
# if the means were actually equal we can't reject the null based on this

z = a_b_test_statistic(1000, 200, 1000, 150)        # -2.94
two_sided_p_value(z)                                # 0.003

# Saying that there is a 0.3% chance we see this difference if it the means
# were actually the same, we can reject the null with this

#Bayesian Inference

# We choose a prior distrubtion from a beta dist and then use obseved data to
# update that dist, if we used our coin as an example we would choose a dist 
# that reprsents whether we think the coin is biased, fair, or we think nothing
# at all then after gathering data we would update that dist, after some tedious
# math we get that the update would simply be alpha + heads, beta + tails

def B(alpha: float, beta: float) -> float:
    """A normalizing constant so that the probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
         return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
