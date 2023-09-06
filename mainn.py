from GPTL import Simulation
import os
import de_simple
name=os.path.basename(__file__).split(".")[0]
file_name = "data/"+str(name)+".csv"
# file_name = "GpMulti0.csv"
util = [0.75,0.85, 0.95]
obj = ["Max-FlowTime", "Mean-FlowTime", "Mean-Weighted-FlowTime"]
sim = Simulation(warm_up_J = 1000,num_M = 10,num_J = 6000,due_date_factor = 4.0,
                 util_level = util[2],Seed = 0,rotate = 10000,Objective = obj[1],
                 pop_num = 1,pop_size = 15,elites = 10,cxPb = 0.8,mutPb = 0.15,
                 NGEN = 2,minPT = 1,maxPT = [49,99],file_name = file_name)
# st = time.perf_counter()
# sim.main_evolution()
# et = time.perf_counter()
# print("time",et-st)

def new_features(vectors,features,args):
    newf = []
    for i in range(len(args)):
        name = args[i]
        if name == 'NIQ':
            val = features[0]
        elif name == 'WIQ':
            val = features[1]
        elif name == 'MWT':
            val = features[2]
        elif name == 'PT':
            val = features[3]
        elif name == 'NPT':
            val = features[4]
        elif name == 'OWT':
            val = features[5]
        elif name == 'W':
            val = features[6]
        elif name == 'NOR':
            val = features[7]
        elif name == 'WKR':
            val = features[8]
        elif name == 'TIS':
            val = features[9]
        newf.append(val * vectors[i])
    return newf

limit = (0,10)
# best_inds = sim.multitask_set()
ind = sim.random_ind()
args = sim.count_param(ind)
func = sim.compile_tree(ind,sim.pset,args)
features = [1,2,3,1,2,3,1,2,3,1]
num = len(args)
vet = [1 for _ in range(num)]
new = new_features(vet,features,args)
fit = func(new)
bounds = [limit for _ in range(num)]

# for i in range(num):
#     bounds.append(limit)
print(bounds)
print(str(ind))
print(args)


popsize = 30                        # population size, must be >= 4
mutate = 0.5                        # mutation factor [0,2]
recombination = 0.7                 # recombination rate [0,1]
maxiter = 50
# de_simple.minimize(sphere, bounds, popsize, mutate, recombination, maxiter)