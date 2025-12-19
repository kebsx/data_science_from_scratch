import enum, random

# An enum is atyped set of enumerated values. We can use them
# to make our code more descriptive and readable
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()

    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

# print("P(both | older):", both_girls / older_girl)
# print("P(both | other):", both_girls / either_girl)

# Uniform distrubtion
def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <= x"""
    if x < 0:   return 0        # uniform random is never less than 0
    elif x < 1: return x        # e.g. P(X <= 0.4) = 0.4
    else:       return 1        # uniform random is always less than 1

import matplotlib.pyplot as plt

xs = []
ys = []
for i in range(0,4):
    xs.append(i - 1)
    ys.append(uniform_cdf(i - 1))

plt.plot(xs, ys)
plt.title("The uniform cdf")
plt.axis([-1, 2, -0.5, 1.5])
# plt.show()

# plt.savefig("images/uniform_cdf.png")
plt.gca().clear()

# The normal distrubtion
# Dependent on the mean (mu) and std (sigma), mean is where bell is centered
# std is how wide the curve is

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')

# Asthetics
plt.legend()
plt.title("Various Normal pdfs")
# plt.show()

# plt.savefig("images/normal_pdfs.png")
plt.gca().clear()


# X = sigma(X) + mu where Z is standard normal variable
# Conversely, Z = (X - mu)/sigma when X is also a standard normal variable

# For the CDF of normal dist we have to use math.erf which is 
# erf(z) = (2/sqrt(pi)) * integral(e^-t^2) 0-z

# Normal CDF
def noraml_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [noraml_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [noraml_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [noraml_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [noraml_cdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')

plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
# plt.show()

plt.savefig("images/normal_cdfs.png")
plt.gca().clear()