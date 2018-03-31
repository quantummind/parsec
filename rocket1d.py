import math
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

elevation = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 22000, 24000, 26000, 28000, 30000])
kinematicViscosity = np.array([1.461E-5, 1.520E-5, 1.581E-5, 1.646E-5, 1.715E-5, 1.787E-5, 1.863E-5, 1.943E-5, 2.028E-5, 2.117E-5, 2.211E-5, 2.311E-5, 2.416E-5, 2.528E-5, 2.646E-5, 2.771E-5, 2.904E-5, 3.046E-5, 3.196E-5, 3.355E-5, 3.525E-5, 3.706E-5, 3.899E-5, 4.213E-5, 4.557E-5, 4.930E-5, 5.333E-5, 5.768E-5, 6.239E-5, 6.749E-5, 7.300E-5, 7.895E-5, 8.540E-5, 9.237E-5, 9.990E-5, 10.805E-5, 11.686E-5, 12.639E-5, 13.670E-5, 14.784E-5, 15.989E-5, 22.201E-5, 30.743E-5, 42.439E-5, 58.405E-5, 80.134E-5])
density = 1.225 * np.array([1.0000, 0.9529, 0.9075, 0.8638, 0.8217, 0.7812, 0.7423, 0.7048, 0.6689, 0.6343, 0.6012, 0.5694, 0.5389, 0.5096, 0.4817, 0.4549, 0.4292, 0.4047, 0.3813, 0.3589, 0.3376, 0.3172, 0.2978, 0.2755, 0.2546, 0.2354, 0.2176, 0.2012, 0.1860, 0.1720, 0.1590, 0.1470, 0.1359, 0.1256, 0.1162, 0.1074, 0.09930, 0.09182, 0.08489, 0.07850, 0.07258, 0.05266, 0.03832, 0.02797, 0.02047, 0.01503])
sound = np.array([340.3, 338.4, 336.4, 334.5, 332.5, 330.6, 328.6, 326.6, 324.6, 322.6, 320.5, 318.5, 316.5, 314.4, 312.3, 310.2, 308.1, 306.0, 303.8, 301.7, 299.8, 297.4, 295.2, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 295.1, 296.4, 297.7, 299.1, 300.4, 301.7])
g = 9.79569		# in Pasadena

radius = 0.1
area = math.pi*(radius)**2
dryMass = 38.0	# kg
massLoss = 1.27724	# kg/s
constThrust = 3558.58		# N, simplified thrust profile
totalImpulse = 40960.0	# N s

# calculates the area based on the radius
def updateArea():
	global area
	area = math.pi*(radius)**2

# returns drag from atmoshpere
def drag(velocity, height):
	rho = np.interp(height, elevation, density)
	mach = velocity / np.interp(height, elevation, sound)
	dragCoeff = np.interp(mach, dragMachs, dragCoeffs)
	# add interpolation for dragCoeff
	
	f_d = 0.5 * dragCoeff * rho * velocity*velocity * area
	return -np.sign(velocity)*f_d

# assumes fuel is being burnt off linearly
def mass(t):
	fuelMass = burnTime()*massLoss
	mass = dryMass + fuelMass
	if t < burnTime():
		return mass - (massLoss*t)
	else:
		return dryMass

# constant thrust profile while fuel is still available
def thrustProfile(t):
	if t < burnTime():
		return constThrust
	else:
		return 0

# assumes fuel is being burnt off linearly
def burnTime():
	return totalImpulse/constThrust

# solve differential equation
def d(z, t):
	thrust = thrustProfile(t)
	m = mass(t)
	return np.array((thrust/m - g + drag(z[0], z[1])/m, z[0]))

def d2(t, z):
	thrust = thrustProfile(t)
	m = mass(t)
	return [z[1], thrust/m - g + drag(z[1], z[0])/m]


def simulate(dfw):
	y0 = 0
	v0 = 0
	z0 = [v0, y0]
	t = np.linspace(0, 500, 100001)
	v, y = odeint(d, z0, t).T

	endInd = np.argmax(y < 0)
	return (np.amax(y)*3.28, dfw.append({'Max Height (ft)':np.amax(y)*3.28, 'Radius (m)':radius, 'Dry Mass (kg)':dryMass, 'Max Velocity (m/s)':np.amax(np.absolute(v[:endInd]))}, ignore_index=True))
	
# 	print('Max height: {} m'.format(np.amax(y)))
# 	print('Max velocity: {} m/s'.format(np.amax(np.absolute(v))))
# 	print('Time to ground: {} s'.format(t[endInd]))
	
# 	plt.ylabel('y (m)')
# 	plt.xlabel('t (s)')
# 	plt.title('Position')
# 	plt.plot(t[:endInd], y[:endInd], label='drag')
# 	plt.figure()
# 	
# 	plt.ylabel('v (m/s)')
# 	plt.xlabel('t (s)')
# 	plt.title('Velocity')
# 	plt.plot(t[:endInd], v[:endInd], label='drag')
# 	plt.show()

dragCoeffs = np.zeros(0)
dragMachs = np.zeros(0)

dryMasses = np.linspace(25, 40, num=16)
thrusts = [3.56]
massLosses = [1.28]
thrusts = [1.01085, 1.21302, 1.41519, 1.61736, 1.81953, 2.02171, 2.22388, 2.42605, 2.62822, 2.83089, 3.03256, 3.23473, 3.4369, 3.63907, 3.84124, 4.04341, 4.24558, 4.44775, 4.64992, 4.85209, 5.05426]
massLosses = [0.36532, 0.43838, 0.51145, 0.58451, 0.65001, 0.73064, 0.8037, 0.87677, 0.94983, 1.0229, 1.09596, 1.16902, 1.24209, 1.31515, 1.38822, 1.46218, 1.53434, 1.60741, 1.68047, 1.75354, 1.8266]
radii = np.linspace(0.06, 0.15, 10)

directory = 'rocket-data'
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.endswith('.csv'):
		df = pd.read_csv(os.path.join(directory, filename))
		dfw = pd.DataFrame()
		dragMachs = df['mach']
		dragCoeffs = df['cd']
		print('Starting file', filename)
		ind = 0
		results = np.empty([radii.size, dryMasses.size])
		for r in range(radii.size):
			radius = radii[r]
			updateArea()
			for m in range(dryMasses.size):
				dryMass = dryMasses[m]
				out = simulate(dfw)
				results[r][m] = out[0]
				dfw = out[1]
			ind += 1
			print(ind/len(radii))
		ax = sns.heatmap(results, center=45000, xticklabels=dryMasses, yticklabels=radii)
		plt.show()
		dfw.to_csv(os.path.join('rocket-output', filename))