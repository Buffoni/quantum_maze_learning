import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#debugFilename = 'debug-dt1-p0'
#debugFilename = 'debug-dt1-p0.001'
#debugFilename = 'debug-dt0.01-p0'
debugFilename = 'debug-dt0.01-p0.001'

df = pd.read_csv(debugFilename+'.csv', header=None)
#print(df.tail())

# s = df.loc[0]
# print(s)
# fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)
# x = s.index
# v = s.values
# ax.bar(x=x, height=v);
# plt.show()

fig, ax = plt.subplots(figsize=(16, 4), dpi=144)

def init():
    ax.clear()
    ax.set_ylim(0., 1.)

def animate(i):
    s = df.loc[i]
    x = s.index
    v = s.values
    for bar in ax.containers:
        bar.remove()
    ax.bar(x=x, height=v, color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=65)
    ax.set_title(f'Step - {i+1}', fontsize='smaller')

ani = FuncAnimation(fig=fig, func=animate, frames=len(df), interval=500, init_func=init, repeat=True)

plt.show()
#ani.save(debugFilename+'.mp4')

