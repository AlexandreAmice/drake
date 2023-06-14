import os
import matplotlib.pyplot as plt
import numpy as np
from visibility_seeding import VisSeeder
import pickle
from seeding_utils import sorted_vertices
import shapely
from shapely.ops import cascaded_union
from pydrake.all import VPolytope
import time


class Logger:
    def __init__(self, world, world_name, seed, N, alpha, eps):
        root = "/home/peter/git/visiris"
        self.world = world
        self.timings = []
        self.name_exp ="experiment_" +world_name+f"_{seed}_{N}_{alpha:.3f}_{eps:.3f}"
        self.expdir = root+"/logs/"+self.name_exp
        self.summary_file = self.expdir+"/summary/summary_"+self.name_exp+".txt"
        M = int(np.log(alpha)/np.log(1-eps) + 0.5)
        if not os.path.exists(self.expdir):
            os.makedirs(self.expdir+"/images")
            os.makedirs(self.expdir+"/data")
            os.makedirs(self.expdir+"/summary")
            with open(self.summary_file, 'w') as f:
                f.write("summary "+self.name_exp+"\n")
                f.write(f"GuardInsertion attempts M:{M}\n")
                f.write(f"{1-alpha:.2f} probability that unseen region is less than {100*eps:.1f} '%' of Cfree ")
                f.write(f"-------------------------------------------\n")           
                f.write(f"-------------------------------------------\n")           
            print('logdir created')
        else:
            print('logdir exists')
    
    def time(self,):
        self.timings.append(time.time())

    def log(self, vs: VisSeeder, iteration):
        #self.timings.append(time.time())
        t_sample = self.timings[-4] - self.timings[-5] 
        t_visgraph = self.timings[-3] - self.timings[-4] 
        t_mhs = self.timings[-2] - self.timings[-3] 
        t_regions = self.timings[-1] - self.timings[-2] 
        t_step = t_sample+t_visgraph + t_mhs + t_regions
        
        t_total = self.timings[-1] - self.timings[0]
        
        #accumulate data
        region_groups_A = [[r.A() for r in g] for g in vs.region_groups] 
        region_groups_b = [[r.b() for r in g] for g in vs.region_groups]
        data = {
        'vg':vs.vgraph_points,
        'vad':vs.vgraph_admat,
        'sp':vs.seed_points,
        'ra':region_groups_A,
        'rb':region_groups_b,
        'tstep':t_step,
        'tsample':t_sample,
        'tsample':t_visgraph,
        'tsample':t_mhs,
        'tsample':t_regions,
        'ttotal':t_total,
        }
        with open(self.expdir+f"/data/it_{iteration}"+".pkl", 'wb') as f:
            pickle.dump(data,f)

        #write summary
        shapely_regions = []
        for r in vs.regions:
            verts = sorted_vertices(VPolytope(r))
            shapely_regions.append(shapely.Polygon(verts.T))
        union_of_Polyhedra = cascaded_union(shapely_regions)
        coverage_experiment = union_of_Polyhedra.area/self.world.cfree_polygon.area
        summary=[f"-------------------------------------------\n",
                 f"ITERATION: {iteration}\n"
                 f"number of regions step {len(vs.region_groups[-1])}\n",
                 f"number of regions total {len(vs.regions)}\n"
                 f"tstep {t_step:.3f}, t_total {t_total:.3f}\n"
                 f"tsample {t_sample:.3f}, t_visgraph {t_visgraph:.3f}, t_mhs {t_mhs:.3f}, t_regions {t_regions:.3f}\n"
                 f"number of regions total {len(vs.regions)}\n"
                 f"coverage {coverage_experiment:.4f}\n"]
        
        with open(self.summary_file, 'a') as f:
            for l in summary:
                f.write(l)
        
        #save picture
        fig, ax = plt.subplots(figsize = (10,10))
        self.world.plot_cfree(ax)
        for g in vs.region_groups:
            rnd_artist = ax.plot([0,0],[0,0], alpha = 0)
            for r in g:
                self.world.plot_HPoly(ax, r, color =rnd_artist[0].get_color())
        ax.set_title(f"iteration {iteration}")
        plt.savefig(self.expdir+f"/images/img_it{iteration}.png")