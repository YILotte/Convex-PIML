import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
# Define and calculate constants and parameters

#Viscosity
nu = 0.1  # enter a number between 0.003 and 0.3 (to maintain numerical stability)

xmax = 5  # Total size
tmax = 1  # Total time

nx = 30  # Number of spatial grid points
dt = 0.04  # Temporal resolution

x = np.linspace(0, xmax, nx)  # Position vector
nt = int(tmax / dt)  #  Number of temporal grid points
dt = tmax / nt  # Temporal resolution (re-calculate for consistancy)
dx = xmax / nx  # Spatial resolution
t = np.linspace(0, tmax, nt)  # Vector containing time points
u = np.zeros([nx, nt])  # Array containing velocity
u[:, 0] = 2 * np.exp(-2 * (x - 0.5 * xmax)**2)  # Initial condition
u[0, :] = 0  # Boundary condition at x=0
u[nx-1, :] = 0  #  Boundary condition at x=xmax

# Time stepping loop for solving the Viscid Burger's equation using finite differences
for n in range(nt-1):
    u[1:nx-1, n+1] = (u[1:nx-1, n] +
                      dt * (nu * (u[2:nx, n] - 2 * u[1:nx-1, n] + u[0:nx-2, n]) / dx**2
                      + 0.25 * (u[2:nx, n]**2 - u[0:nx-2, n]**2) / dx))

# Select time points to plotting velocity vs position
timepoints = np.array([0, tmax / 5, tmax / 3, tmax / 2, tmax/1.1]) / dt
# Generate levels for contour plot
levels = np.arange(0, np.amax(u) + 0.01,  np.amax(u) / 50)
# Create figure
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
# Plot contour (velocity given by colorbar)
cf = ax1[0].contourf(x, t, u.T, levels=levels)
# Add colorbar
cbar = fig1.colorbar(cf, ax=ax1[0])
# Add labels
ax1[0].set_xlabel('Position')
ax1[0].set_ylabel('Time')
# Plot velocity vs position for 5 different time points
ax1[1].plot(x, u[:, int(timepoints[0])], label='t=0')
ax1[1].plot(x, u[:, int(timepoints[1])], label='t=T/5')
ax1[1].plot(x, u[:, int(timepoints[2])], label='t=T/3')
ax1[1].plot(x, u[:, int(timepoints[3])], label='t=T/2')
ax1[1].plot(x, u[:, int(timepoints[4])], label='t=T')
# Add labels
ax1[1].set_xlabel('Position')
ax1[1].set_ylabel('Velocity')
ax1[1].legend()
fig1.tight_layout()
# show figure
plt.show()

# Save data
data=np.zeros((1,u.shape[0]*u.shape[1]))
for i in range(0,u.shape[0]):
    data[0,i*u.shape[1]:(i+1)*u.shape[1]]=u[i,:]
data=data.reshape(data.shape[1],)

x1=x.repeat(len(t))
t1=np.tile(t, (1,len(x)))
t1=t1.reshape(t1.shape[1],)

df=pd.DataFrame({'x':x1,
                 't':t1,
                 'u':data})

df.to_csv('burgers.csv',sep=',',index=False,header=True)