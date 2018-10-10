'''
Helper script for converting loss function data into usable graphs.
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pgf.rcfonts'] = False

iters = []
d_loss = []
g_loss = []

with open('logs/testing_baseline.txt') as f:
    for line in f:
        words = line.split(' ')
        iters.append(int(words[0]))
        d_loss.append(float(words[1]))
        g_loss.append(float(words[2]))

plt.plot(iters, g_loss, '-b', label='discriminator loss')
plt.plot(iters, d_loss, '-r', label='generator loss')

plt.xlabel("iteration")
plt.legend(loc='upper left')
plt.title('LOSS')
# save to file
plt.savefig('example.pdf')