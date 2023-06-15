import os
import matplotlib.pyplot as plt
import numpy as np
from visibility_seeding import VisSeeder
import pickle
import time
from datetime import datetime
import networkx as nx
from visibility_utils import generate_distinct_colors

class Logger:
    def __init__(self, experiment_name, seed, N, alpha, eps, estimate_coverage):
        root = "/home/peter/git/drake_visiris_build/drake/C_Iris_Examples/"
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")
        self.timings = []
        self.name_exp =experiment_name + "_" +timestamp_str+f"_{seed}_{N}_{alpha:.3f}_{eps:.3f}"
        self.expdir = root+"/logs/"+self.name_exp
        self.summary_file = self.expdir+"/summary/summary_"+self.name_exp+".txt"
        M = int(np.log(alpha)/np.log(1-eps) + 0.5)
        self.coverage_estimator = estimate_coverage
            
        if not os.path.exists(self.expdir):
            os.makedirs(self.expdir+"/data")
            os.makedirs(self.expdir+"/summary")
            os.makedirs(self.expdir+"/images")
            with open(self.summary_file, 'w') as f:
                f.write("summary "+self.name_exp+"\n")
                f.write(f"Point Insertion attempts M:{M}\n")
                f.write(f"{1-alpha:.2f} probability that unseen region is less than {100*eps:.1f} '%' of Cfree \n")
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
        
        #write summary
        coverage_experiment = self.coverage_estimator(vs.regions)
        
        data = {
        'vg':vs.vgraph_points,
        'vad':vs.vgraph_admat,
        'sp':vs.seed_points,
        'ra':region_groups_A,
        'rb':region_groups_b,
        'cov': coverage_experiment,
        'tstep':t_step,
        'tsample':t_sample,
        'tsample':t_visgraph,
        'tsample':t_mhs,
        'tsample':t_regions,
        'ttotal':t_total,
        }
        with open(self.expdir+f"/data/it_{iteration}"+".pkl", 'wb') as f:
            pickle.dump(data,f)

        summary=[f"-------------------------------------------\n",
                 f"ITERATION: {iteration}\n"
                 f"number of regions step {len(vs.region_groups[-1])}\n",
                 f"number of regions total {len(vs.regions)}\n"
                 f"tstep {t_step:.3f}, t_total {t_total:.3f}\n"
                 f"tsample {t_sample:.3f}, t_visgraph {t_visgraph:.3f}\n"
                 f"t_mhs {t_mhs:.3f}\n"
                 f"t_regions {t_regions:.3f}\n"
                 f"number of regions total {len(vs.regions)}\n"
                 f"coverage {coverage_experiment:.4f}\n"]
        
        with open(self.summary_file, 'a') as f:
            for l in summary:
                f.write(l)
        
        self.connectivity_graph = nx.Graph()
        for idx in range(len(vs.regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(vs.regions)):
            for idx2 in range(idx1 +1, len(vs.regions)):
                r1 = vs.regions[idx1]
                r2 = vs.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)

        fig = plt.figure(figsize=(10,10))
        hues = generate_distinct_colors(len(vs.region_groups))
        colors = []
        for g,h in zip(vs.region_groups, hues):
            colors += [h]*len(g)

        nx.draw_spring(self.connectivity_graph, 
                       with_labels = True, 
                       node_color = colors)
        plt.title(f"iteration {iteration}")
        plt.savefig(self.expdir+f"/images/img_it{iteration}.png")

        # #save picture
        # fig, ax = plt.subplots(figsize = (10,10))
        # self.world.plot_cfree(ax)
        # for g in vs.region_groups:
        #     rnd_artist = ax.plot([0,0],[0,0], alpha = 0)
        #     for r in g:
        #         self.world.plot_HPoly(ax, r, color =rnd_artist[0].get_color())
        # ax.set_title(f"iteration {iteration}")
        # plt.savefig(self.expdir+f"/images/img_it{iteration}.png")