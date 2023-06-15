import numpy as np
from pydrake.all import HPolyhedron
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def shrink_regions(regions, offset_fraction = 0.25):
	shrunken_regions = []
	for r in regions:
		offset = offset_fraction*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
		rnew = HPolyhedron(r.A(), r.b()-offset)
		shrunken_regions.append(rnew)	
	return shrunken_regions

def point_in_regions(pt, regions):
    for r in regions:
        if r.PointInSet(pt.reshape(-1,1)):
            return True
    return False

def generate_distinct_colors(n):
    cmap = plt.cm.get_cmap('hsv', n)  # Choose a colormap
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]  # Convert colormap to hexadecimal colors
    return colors