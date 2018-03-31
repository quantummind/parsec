import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# up is positive

g = 9.79569		# g in Pasadena
radius = 0.05
mass0 = 1
mass = mass0
# https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html
elevation = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 22000, 24000, 26000, 28000, 30000])
kinematicViscosity = np.array([1.461E-5, 1.520E-5, 1.581E-5, 1.646E-5, 1.715E-5, 1.787E-5, 1.863E-5, 1.943E-5, 2.028E-5, 2.117E-5, 2.211E-5, 2.311E-5, 2.416E-5, 2.528E-5, 2.646E-5, 2.771E-5, 2.904E-5, 3.046E-5, 3.196E-5, 3.355E-5, 3.525E-5, 3.706E-5, 3.899E-5, 4.213E-5, 4.557E-5, 4.930E-5, 5.333E-5, 5.768E-5, 6.239E-5, 6.749E-5, 7.300E-5, 7.895E-5, 8.540E-5, 9.237E-5, 9.990E-5, 10.805E-5, 11.686E-5, 12.639E-5, 13.670E-5, 14.784E-5, 15.989E-5, 22.201E-5, 30.743E-5, 42.439E-5, 58.405E-5, 80.134E-5])
density = 1.225 * np.array([1.0000, 0.9529, 0.9075, 0.8638, 0.8217, 0.7812, 0.7423, 0.7048, 0.6689, 0.6343, 0.6012, 0.5694, 0.5389, 0.5096, 0.4817, 0.4549, 0.4292, 0.4047, 0.3813, 0.3589, 0.3376, 0.3172, 0.2978, 0.2755, 0.2546, 0.2354, 0.2176, 0.2012, 0.1860, 0.1720, 0.1590, 0.1470, 0.1359, 0.1256, 0.1162, 0.1074, 0.09930, 0.09182, 0.08489, 0.07850, 0.07258, 0.05266, 0.03832, 0.02797, 0.02047, 0.01503])
# pressure = np.array([1.01325E5, 0.9546E5, 0.8988E5, 0.8456E5, 0.7950E5, 0.7469E5, 0.7012E5, 0.6578E5, 0.6166E5, 0.5775E5, 0.5405E5, 0.5054E5, 0.4722E5, 0.4408E5, 0.4111E5, 0.3830E5, 0.3565E5, 0.3315E5, 0.3080E5, 0.2858E5, 0.2650E5, 0.2454E5, 0.2270E5, 0.2098E5, 0.1940E5, 0.1793E5, 0.1658E5, 0.1533E5, 0.1417E5, 0.1310E5, 0.1211E5, 0.1120E5, 0.1035E5, 0.09572E5, 0.08850E5, 0.08182E5, 0.07565E5, 0.06995E5, 0.06467E5, 0.05980E5, 0.05529E5, 0.04047E5, 0.02972E5, 0.02188E5, 0.01616E5, 0.01197E5])

# http://web.hallym.ac.kr/~physics/education/Math/calculus/drag/drag7.html
reynolds = np.array([0.05875, 0.1585, 0.4786, 3.020, 7.015, 15.49, 57.54,144.5, 264.9, 512.9, 1000, 1862, 3162, 4764, 8375, 0.1556E5, 0.2648E5, 0.3467E5, 0.5888E5, 0.1000E6, 0.1702E6, 0.2317E6, 0.2648E6, 0.2710E6, 0.2851E6, 0.3020E6, 0.3388E6, 0.3981E6, 0.5129E6, 0.1778E7, 0.2291E7, 0.5012E7])
dragCoeff = np.array([492.0, 169.8, 58.88, 10.86, 5.623, 3.388, 1.479, 0.9204, 0.7194, 0.5623, 0.4786, 0.4365, 0.4074, 0.3890, 0.3981, 0.4395, 0.4571, 0.4775, 0.4732, 0.4624, 0.4395, 0.4046, 0.3733, 0.3467, 0.2472, 0.1778, 0.1047, 0.09772, 0.1000, 0.1778, 0.1862, 0.1862])


def drag(velocity, height):
	area = np.pi * radius**2
	re = abs(velocity*2*radius/np.interp(height, elevation, kinematicViscosity))
	c_d = log_interp(re, reynolds, dragCoeff)
	rho = np.interp(height, elevation, density)
	
	f_d = 0.5 * c_d * rho * velocity*velocity * area
	return -np.sign(velocity)*f_d
	
def log_interp(xVal, x, y):
	return np.power(10.0, np.interp(np.log10(xVal), np.log10(x), np.log10(y)))

def spring(height, velocity):
	global mass
	springLen = 5
	k = 50
	
	if height < springLen and velocity > 0:
		mass = 0.5*mass0*(springLen - height)/springLen + 0.5
		return -k*(height-springLen)
	else:
		return 0

def d(z, t):
	return np.array((spring(z[1], z[0]) - mass*g + drag(z[0], z[1]), z[0]))
def dIdeal(z, t):
	return np.array((spring(z[1], z[0]) - mass*g, z[0]))

def simulate():
	y0 = 0.0000001
	v0 = 0.0000001
	z0 = [v0, y0]
	t = np.linspace(0, 15, 1000000)
	v, y = odeint(d, z0, t).T
	vIdeal, yIdeal = odeint(dIdeal, z0, t).T
	endInd = np.argmax(y < 0)
	endIndIdeal = np.argmax(yIdeal < 0)
	
	print('DRAG')
	print('Max height: {} m'.format(np.amax(y)))
	print('Max velocity: {} m/s'.format(np.amax(np.absolute(v))))
	print('Time to ground: {} s'.format(t[endInd]))
	print('\nNO DRAG')
	print('Max height: {} m'.format(np.amax(yIdeal)))
	print('Max velocity: {} m/s'.format(np.amax(np.absolute(vIdeal))))
	print('Time to ground: {} s'.format(t[endIndIdeal]))
	
	plt.ylabel('y (m)')
	plt.xlabel('t (s)')
	plt.title('Position')
	y1, = plt.plot(t[:endInd], y[:endInd], label='drag')
	y2, = plt.plot(t[:endIndIdeal], yIdeal[:endIndIdeal], label='no drag')
	plt.legend(handles=[y1, y2])
	plt.figure()
	
	plt.ylabel('v (m/s)')
	plt.xlabel('t (s)')
	plt.title('Velocity')
	v1, = plt.plot(t[:endInd], v[:endInd], label='drag')
	v2, = plt.plot(t[:endIndIdeal], vIdeal[:endIndIdeal], label='no drag')
	plt.legend(handles=[v1, v2])
	plt.show()

simulate()