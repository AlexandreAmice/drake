import numpy as np
from pydrake.all import HPolyhedron

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
