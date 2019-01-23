'''
Helper script for converting loss function data into simple graphs.
log data should be a simple text file where each line is of format:
[iteration number] [discriminator loss] [generator loss]
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pgf.rcfonts'] = False

iters = []
d_loss = []
g_loss = []

with open('logs/dummy_logs.txt') as f:
    for line in f:
        words = line.split(' ')
        iters.append(int(words[0]))
        d_loss.append(float(words[1]))
        g_loss.append(float(words[2]))

plt.plot(iters, d_loss, '-b', label='discriminator loss')
plt.plot(iters, g_loss, '-r', label='generator loss')

plt.xlabel("iteration")
plt.legend(loc='lower right')
plt.title('NasNet Mobile loss')
# save to file
#plt.savefig('report_images/nasnet_loss.pdf')
plt.show()