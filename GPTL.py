#coding:utf-8
import queue
import numpy as np
import statistics
import math
import operator
from collections import defaultdict, deque
import random
# from gantt import *
from deap import algorithms,base,creator,gp,tools
import copy
import time
import csv
import os
from collections import namedtuple
from scipy.stats import rankdata
import re
from lshash.lshash import LSHash

class Simulation:
    def __init__(self,warm_up_J,num_M,num_J,due_date_factor,util_level,Seed,rotate,Objective,pop_num,pop_size,elites,cxPb,mutPb,NGEN,minPT,maxPT:list,file_name):
        self.warm_up_J = warm_up_J
        self.num_M = num_M
        self.num_J = num_J
        self.due_date_factor = due_date_factor
        self.util_level = util_level
        self.Seed = Seed
        self.rotate = rotate
        self.Objective = Objective
        self.minPT = minPT
        self.maxPT = maxPT
        #job param
        self.num_ops = np.zeros(self.num_J, dtype=int)
        self.num_options = [0] * self.num_J
        self.op_pt_time = [0.0] * self.num_J
        self.weights = np.zeros(self.num_J, dtype=int)
        self.arrival_time = np.zeros(self.num_J)
        self.m_of_op = [0] * self.num_J
        self.pt_of_op = [0.0] * self.num_J
        self.median_proc_time = [[] for _ in range(self.num_J)]
        self.total_proc_time = [0.0] * self.num_J
        self.lamda = self.arrival_interval_lamda(self.minPT,self.maxPT[1],0.85)
        self.arrival_interval = np.random.exponential(self.lamda, size=self.num_J)
        self.arrival_time[0] = self.arrival_interval[0]
        # simulation param
        self.m_first_work = np.zeros(self.num_M, dtype=bool)  # 检测机器是否第一次激活
        self.pt_record = np.zeros((self.num_J, max(self.num_ops)))  # pt_record要放在工件属性生成之后
        self.op_process = np.ones(self.num_J, dtype=int)
        self.completion_time = np.zeros(self.num_J)
        self.due_date = np.zeros(self.num_J)
        self.flow_t = np.zeros(self.num_J)
        self.w_flow_t = np.zeros(self.num_J)
        # median_pt = [[] for _ in range(num_J)]
        self.op_done_t = np.zeros((self.num_J, max(self.num_ops)))
        self.m_done_t = np.zeros(self.num_M)  # 只记录m最新一次完成工作的时间
        self.niq = np.zeros(self.num_M, dtype=int)  # 激活成功后数量+1
        self.wiq = np.zeros(self.num_M)  # 激活成功后时间+pt
        self.mwt = np.zeros(self.num_M)  # +wait_t:指定m_id:activate - work_done_t(机器回到等待队列的时间)
        self.npt = np.zeros(self.num_J)  # npt[j_id - 1] = op_median_pt[j_id - 1][op_id]
        self.wkr = np.zeros(self.num_J)
        self.nor = np.zeros(self.num_J, dtype=int)
        self.owt = np.zeros(
            (self.num_J, max(self.num_ops)))  # op待在J_pool的时间：activate - 上一个op完成时间,第一个op为activate - arrival
        self.J_pool = []
        self.work_done_t = []
        self.machines_wait = [i + 1 for i in range(self.num_M)]
        self.machine_queue = [[] for _ in range(self.num_M)]
        self.record_machine_queue = [[] for _ in range(self.num_M)]
        self.work_m = [[] for _ in range(self.num_M)]
        self.features_storage = []
        #init pop
        self.gen = 0
        self.pop_num = pop_num
        self.popSize = pop_size
        self.elites = elites
        self.cxPb = cxPb
        self.mutPb = mutPb
        self.NGEN = NGEN
        self.DC = []
        self.DC_rank = []
        self.PC = [[] for _ in range(self.popSize)]
        self.toolbox = base.Toolbox()
        self.pset = self.get_pset()
        self.compile_individual(self.pset, self.toolbox)
        if pop_num > 1:
            # self.best_ind = 0
            # self.func_c = 0
            self.subpop = [0.0 for i in range(self.popSize)]
            for i in range(self.pop_num):
                Subpop = self.toolbox.population(n=self.popSize)
                self.subpop[i] = Subpop  # a population with 1024 individuals
                self.register_gp_operators(self.toolbox, self.pset,Subpop)
            self.best_ind = self.subpop[0][0]
        else:
            self.pop = self.toolbox.population(n=self.popSize)
            self.register_gp_operators(self.toolbox, self.pset,self.pop)
        self.file_name = file_name

    def protectedDiv(self,left, right):
        if right != 0:
            return left / right
        else:
            return 1

    def get_pset(self):
        pset = gp.PrimitiveSet('MAIN', 10)
        pset.addPrimitive(max, 2)
        pset.addPrimitive(min, 2)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(self.protectedDiv, 2)
        pset.renameArguments(ARG0='NIQ')
        pset.renameArguments(ARG1='WIQ')
        pset.renameArguments(ARG2='MWT')
        pset.renameArguments(ARG3='PT')
        pset.renameArguments(ARG4='NPT')
        pset.renameArguments(ARG5='OWT')
        pset.renameArguments(ARG6='W')
        pset.renameArguments(ARG7='NOR')
        pset.renameArguments(ARG8='WKR')
        pset.renameArguments(ARG9='TIS')
        return pset

    def compile_individual(self,pset, toolbox):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

    def register_gp_operators(self,toolbox, pset,pop):
        limitHeight = 8
        # toolbox.register('evaluate', evalTree,toolbox = toolbox)
        # toolbox.register('evaluate', eval)
        # toolbox.register('priority', self.get_priority, features=self.features_storage,func=self.func)
        toolbox.register('evaluate', self.set_fitness, pop=pop, objective=self.Objective, flow_t=self.flow_t, w_flow_t=self.w_flow_t)
        toolbox.register('select', tools.selTournament, tournsize=7)
        toolbox.register('mate', gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.register("expr_mut", gp.genGrow, min_=4, max_=4)
        toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight))

    def phenotypic_characterise(self,referRule1:int,referRule2:int,features,able_num,func):
        #referRule1--WIQ,referRule2--SPT ---> -(PT + WIQ + NIQ)
        tmp = np.array(features)
        tmp1 = tmp[:,referRule1]
        tmp2 = tmp[:,referRule2]
        tmp3 = tmp[:,0]#NIQ
        refer_value = -(np.add(tmp1,tmp2,tmp3))
        rank = rankdata(refer_value, method='min')
        fit_l = []
        for f in range(len(features)):
            fit = func(features[f][0], features[f][1], features[f][2], features[f][3], features[f][4], features[f][5],
                       features[f][6], features[f][7], features[f][8], features[f][9])
            fit_l.append(fit)
        rank_c = list(rankdata(np.array(fit_l), method='min'))
        pc = rank[rank_c.index(min(rank_c))]
        # count = 1
        # while pc > able_num:
        #     pc = rank[rank_c.index(min(rank_c) + count)]
        #     count += 1
        return pc

    def save_pc(self,ind,pc):
        # if len(self.PC[ind]) == 0:
        #     self.PC[ind][0] = pc
        # else:
        self.PC[ind].append(pc)

    def get_priority(self,features,func):
        min_j = 0
        fit_l = []
        min_fit = math.inf
        for f in range(len(features)):
            fit = func(features[f][0], features[f][1], features[f][2], features[f][3], features[f][4], features[f][5],
                       features[f][6], features[f][7], features[f][8], features[f][9])
            fit_l.append(fit)
            if fit < min_fit:
                min_fit = fit
                min_j = f
        fit_same_J = np.where(fit_l == min_fit)
        if len(fit_same_J) > 1:
            min_j = np.random.choice(fit_same_J)
        return min_j

    # "Max-FlowTime", "Mean-FlowTime", "Mean-Weighted-FlowTime"
    def set_fitness(self,pop, objective, flow_t, w_flow_t, p):
        if objective == "Max-FlowTime":
            pop[p].fitness.values = [max(flow_t[self.warm_up_J:])]
        elif objective == "Mean-FlowTime":
            pop[p].fitness.values = [statistics.mean(flow_t[self.warm_up_J:])]
        elif objective == "Mean-Weighted-FlowTime":
            pop[p].fitness.values = [statistics.mean(w_flow_t[self.warm_up_J:])]

    def set_fitness_change(self,pop, objective, flow_t, w_flow_t, p, task_num,task_no):
        if task_num == 1:
            fit = self.cal_fitness(objective, flow_t, w_flow_t)
            pop[p].fitness.values = [fit]
        else:
            l = [0.0]* task_num
            pop[p].fitness.values = l
            fit = self.cal_fitness()
            pop[p].fitness.values[task_no] = fit

    def cal_fitness(self,objective, flow_t, w_flow_t):
        if objective == "Max-FlowTime":
            fit = max(flow_t[self.warm_up_J:])
        elif objective == "Mean-FlowTime":
            fit = statistics.mean(flow_t[self.warm_up_J:])
        elif objective == "Mean-Weighted-FlowTime":
            fit = statistics.mean(w_flow_t[self.warm_up_J:])
        return fit

    def init_job_param(self):
        np.random.seed(self.Seed)
        self.num_ops = np.zeros(self.num_J, dtype=int)
        self.num_options = [0] * self.num_J
        self.op_pt_time = [0.0] * self.num_J
        self.due_date = np.zeros(self.num_J)
        self.weights = np.zeros(self.num_J, dtype=int)
        self.arrival_time = np.zeros(self.num_J)
        self.m_of_op = [0] * self.num_J
        self.pt_of_op = [0.0] * self.num_J
        self.median_proc_time = [[] for _ in range(self.num_J)]
        self.total_proc_time = [0.0] * self.num_J
        self.lamda = self.arrival_interval_lamda(self.minPT,self.maxPT[1],0.85)
        self.arrival_interval = np.random.exponential(self.lamda, size=self.num_J)
        self.arrival_time[0] = self.arrival_interval[0]
        self.PC = [[]for _ in range(self.popSize)]
        self.DC = []
        self.DC_rank = []

    def init_job_param_for_task(self):
        np.random.seed(self.Seed)
        self.num_ops = np.zeros(self.num_J, dtype=int)
        self.num_options = [0] * self.num_J
        self.op_pt_time = [0.0] * self.num_J
        self.due_date = np.zeros(self.num_J)
        self.weights = np.zeros(self.num_J, dtype=int)
        self.arrival_time = np.zeros(self.num_J)
        self.m_of_op = [0] * self.num_J
        self.pt_of_op = [0.0] * self.num_J
        self.median_proc_time = [[] for _ in range(self.num_J)]
        self.total_proc_time = [0.0] * self.num_J
        self.PC = [[] for _ in range(self.popSize)]
        self.DC = []
        self.DC_rank = []

    def init_sim_param(self):
        self.m_first_work = np.zeros(self.num_M, dtype=bool)  # 检测机器是否第一次激活
        self.pt_record = np.zeros((self.num_J, max(self.num_ops)))  # pt_record要放在工件属性生成之后
        self.op_process = np.ones(self.num_J, dtype=int)
        self.completion_time = np.zeros(self.num_J)
        self.flow_t = np.zeros(self.num_J)
        self.w_flow_t = np.zeros(self.num_J)
        # median_pt = [[] for _ in range(num_J)]
        self.op_done_t = np.zeros((self.num_J, max(self.num_ops)))
        self.m_done_t = np.zeros(self.num_M)  # 只记录m最新一次完成工作的时间
        self.niq = np.zeros(self.num_M, dtype=int)  # 激活成功后数量+1
        self.wiq = np.zeros(self.num_M)  # 激活成功后时间+pt
        self.mwt = np.zeros(self.num_M)  # +wait_t:指定m_id:activate - work_done_t(机器回到等待队列的时间)
        self.npt = np.zeros(self.num_J)  # npt[j_id - 1] = op_median_pt[j_id - 1][op_id]
        self.wkr = np.zeros(self.num_J)
        self.nor = np.zeros(self.num_J, dtype=int)
        self.owt = np.zeros(
            (self.num_J, max(self.num_ops)))  # op待在J_pool的时间：activate - 上一个op完成时间,第一个op为activate - arrival
        self.J_pool = []
        self.work_done_t = []
        self.machines_wait = [i + 1 for i in range(self.num_M)]
        self.machine_queue = [[] for _ in range(self.num_M)]
        self.record_machine_queue = [[] for _ in range(self.num_M)]
        self.work_m = [[] for _ in range(self.num_M)]
        self.features_storage = []

    def two_six_two_sampler(self):
        value = 1
        r = np.random.uniform(0, 1)
        if r < 0.2:
            value = 4
        elif r < 0.8:
            value = 2
        return value

    def randint_size_n(self,n, N):  # 使用randint随机选择n个数
        return np.random.randint(1, N, size=n)

    def generate_jobs_attribute_use_np(self,max_pt):
        num_operations = np.random.randint(1, 11)
        weight = self.two_six_two_sampler()
        options = np.zeros(self.num_J, dtype=int)
        m_of_op = [0] * num_operations
        pt_of_op = [0.0] * num_operations
        for i in range(num_operations):
            # candidate_machines_of_op = []#每一个operation清空一次列表
            # op_option_dict = {}
            num_options = np.random.randint(1, self.num_M + 1)
            options[i] = num_options
            m = np.arange(1, self.num_M + 1)
            # candidate_machines_of_op = np.random.choice(m,num_options,replace=False)
            # if num_options <= num_M:
            candidate_machines_of_op = self.randint_size_n(num_options, self.num_M + 1)
            if len(np.where(candidate_machines_of_op)) > 1:
                print('error')
            proc_time = np.random.uniform(1, max_pt + 1, size=[num_options])
            m_of_op[i] = list(candidate_machines_of_op)
            pt_of_op[i] = list(proc_time)
        return num_operations, options, weight, m_of_op, pt_of_op

    def candidate_machine_use_np(self,j_id, op_id):
        needed_machine = self.m_of_op[j_id - 1][op_id - 1]
        needed_pt = self.pt_of_op[j_id - 1][op_id - 1]
        return needed_machine, needed_pt

    def link_operations_use_np(self,num_operations, j_index):
        # work_remaining = 0.0
        # for i in range(len(num_operations)):
        #     num_ops_remaining = num_operations[i] - process_pt[i]
        for j in range(num_operations[j_index]):
            proc_times = self.pt_of_op[j_index][j]
            self.median_proc_time[j_index].append(statistics.median(proc_times))
        self.total_proc_time[j_index] = sum(self.median_proc_time[j_index])

    #EVENT-----------------------------------------------------------------
    def job_arrival_event(self,arrival_time):
        for j in range(self.num_J):
            self.schedule_queue.put([arrival_time[j], 1, "JA" + str(j + 1)])

    def machine_activate_event(self,t):
        # if len(machines_wait) != 0 and len(J_pool) != 0:
        for i in range(len(self.machines_wait)):
            m_id = self.machines_wait[i]
            self.schedule_queue.put([t, 2, "MA" + str(m_id)])

    def machine_work_done_event(self,t, m_id):
        self.schedule_queue.put([t, 3, "MD" + str(m_id)])

    def sim_end_event(self,t):
        self.schedule_queue.put([t, 4, "SE"])

    #EVENT util----------------------------------------------------------
    def activate_info(self,j_id, op_id, m_id, j_index, avail_job_pt, T):
        self.J_pool.remove(j_id)
        pt = avail_job_pt[j_index]
        self.pt_record[j_id - 1][op_id - 1] = pt
        self.update_owt(j_id=j_id, op_id=op_id, at=T)
        self.update_niq_wiq_mwt(pt=pt, m_id=m_id, at=T)
        self.work_m[m_id - 1].append([T, 0.0])
        self.machines_wait.remove(m_id)
        self.machine_queue[m_id - 1].append((j_id, op_id))
        self.record_machine_queue[m_id - 1].append((j_id, op_id))
        self.op_done_t[j_id - 1][op_id - 1] = T + pt
        self.machine_work_done_event(T + pt, m_id)

    def init_features(self):
        for i in range(self.num_J):
            self.nor[i] = self.num_ops[i] - (self.op_process[i] - 1)
            if self.op_process[i] == self.num_ops[i]:
                self.npt[i] = 0.0
            else:
                self.npt[i] = self.median_proc_time[i][self.op_process[i]]
            op_mid = self.median_proc_time[i]
            self.wkr[i] = sum(op_mid[-self.nor[i]:])

    # work_done后调用
    # op_done_t[j_id - 1][op_id - 1] = t,t为op完成时间
    def update_nor_wkr_npt(self,j_id):
        # nor = [num_ops[i]-(op_process[i] - 1) for i in range(num_J)]
        # for i in range(num_J):
        self.nor[j_id - 1] = self.num_ops[j_id - 1] - (self.op_process[j_id - 1] - 1)
        if self.op_process[j_id - 1] == self.num_ops[j_id - 1]:
            self.npt[j_id - 1] = 0.0
        else:
            self.npt[j_id - 1] = self.median_proc_time[j_id - 1][self.op_process[j_id - 1]]
        op_mid = self.median_proc_time[j_id - 1]
        self.wkr[j_id - 1] = sum(op_mid[-self.nor[j_id - 1]:])

    # activate成功调用
    def update_owt(self,j_id, op_id, at):
        if op_id != 1:
            self.owt[j_id - 1][op_id - 1] = at - self.op_done_t[j_id - 1][op_id - 2]
        else:
            self.owt[j_id - 1][op_id - 1] = at - self.arrival_time[j_id - 1]

    # 机器激活成功后调用
    def update_niq_wiq_mwt(self,pt, m_id, at):
        self.niq[m_id - 1] += 1
        self.wiq[m_id - 1] += pt
        if self.m_first_work[m_id - 1] == True:
            self.mwt[m_id - 1] += at - self.m_done_t[m_id - 1]
        else:
            self.mwt[m_id - 1] += at
            self.m_first_work[m_id - 1] = True

    # 用在选工件之前
    def get_pt_feature(self,m_id):
        pt, avail_j = [], []
        for i in range(len(self.J_pool)):
            ms, pts = self.candidate_machine(j_id=self.J_pool[i], op_id=self.op_process[self.J_pool[i] - 1])
            if m_id in ms:
                avail_j.append(self.J_pool[i])
                pt.append(pts[ms.index(m_id)])
        return avail_j, pt

    def avail_job_features_use_np(self,avail_j, m_id, pt, at):
        features = [0.0] * len(avail_j)
        for i in range(len(avail_j)):
            j_id = avail_j[i]
            op_id = self.op_process[j_id - 1]
            features[i] = np.array([self.niq[m_id - 1], self.wiq[m_id - 1], self.mwt[m_id - 1], pt[i], self.npt[j_id - 1],
                                    self.owt[j_id - 1][op_id - 1], self.weights[j_id - 1], self.nor[j_id - 1], self.wkr[j_id - 1], at])
        return features

    def get_due_date(self,j_index):
        # due_date = [0.0] * self.num_J
        due_date = self.arrival_time[j_index] + self.due_date_factor * self.total_proc_time[j_index]
        return due_date

    def get_flow_time(self):
        for i in range(self.num_J):
            self.flow_t[i] = (self.completion_time[i] - self.arrival_time[i])
            self.w_flow_t[i] = (self.weights[i] * self.flow_t[i])
            if self.flow_t[i] < 0 :
                print('flow_t < 0 is impossible')
            if self.w_flow_t[i] < 0:
                print('w_flow_t < 0 is impossible')

    def arrival_interval_lamda(self,min_pt,max_pt,util):
        mean_num_ops = (1 + 10) * 0.5  # minNumOps:1,maxNumOps:10
        mean_pt = (min_pt + max_pt) * 0.5
        lamda = (mean_num_ops * mean_pt) / (util * self.num_M)
        return lamda

    #EVENT SOLUTE
    def check_job_arrival(self,item, task_no):
        T = item
        self.J_pool.append(task_no)  # task_no = job_id
        if len(self.machines_wait) != 0 and len(self.J_pool) != 0:
            self.machine_activate_event(T)

    def check_machine_work_done_event(self,T, task_no):
        m_id = task_no
        t = T
        j_id = self.machine_queue[m_id - 1][0][0]
        self.machine_queue[m_id - 1].pop(0)
        # machines_state[m_id - 1] = True
        self.machines_wait.append(m_id)
        # machines_wait.sort()
        self.work_m[m_id - 1][-1][-1] = t
        self.work_done_t.append(t)
        self.m_done_t[m_id - 1] = t
        self.update_nor_wkr_npt(j_id)
        if self.op_process[j_id - 1] < self.num_ops[j_id - 1]:  # job不是最后一道工序
            self.op_process[j_id - 1] += 1
            self.J_pool.append(j_id)
        elif self.op_process[j_id - 1] == self.num_ops[j_id - 1]:
            self.completion_time[j_id - 1] = t
        if len(self.J_pool) != 0 and len(self.machines_wait) != 0:
            self.machine_activate_event(t)

    def check_machine_activate(self,item, task_no):
        m_id = task_no
        T = item
        end = False
        if m_id in self.machines_wait and len(self.J_pool) > 0:
            avail_job = []
            avail_job_pt = []
            for i in range(len(self.J_pool)):
                j_id = self.J_pool[i]
                op_id = self.op_process[j_id - 1]
                # needed_machine, needed_pt = candidate_machine(j_id, op_id)
                needed_machine, needed_pt = self.candidate_machine_use_np(j_id, op_id)
                if m_id in needed_machine:
                    avail_job.append(j_id)
                    avail_job_pt.append(needed_pt[needed_machine.index(m_id)])
            if len(avail_job) > 100:
                self.sim_end_event(T - 1)
                end = True
            if not end:
                if len(avail_job) == 1:
                    j_index = 0
                    j_id = avail_job[0]
                    op_id = self.op_process[j_id - 1]
                    self.activate_info(j_id, op_id, m_id, j_index, avail_job_pt, T)
                elif len(avail_job) > 1:
                    # if len(avail_job) < 100:
                    for count in range(len(avail_job)):
                        id1 = avail_job[count]
                        id2 = self.op_process[id1 - 1]
                        self.update_owt(j_id=id1, op_id=id2, at=T)
                    features_storage = self.avail_job_features_use_np(avail_job, m_id, avail_job_pt, T)
                    j_index = self.get_priority(features_storage,self.func)
                    # random_bool = random.choice([True, False])
                    # if T > self.arrival_time[999] and len(avail_job) >= 7 and len(self.DC) < 40 and random_bool:
                    #     #每一代都抽取40个决策场景，存features和gp个体计算出的rank
                    #     self.DC.append(features_storage)
                    #     self.DC_rank.append(rank_c)
                    # self.record_DS(avail_job,features_storage,T)
                        # pc = self.phenotypic_characterise(3,1,features_storage,7,rank_c)
                        # self.save_pc(p,pc)
                    j_id = avail_job[j_index]
                    op_id = self.op_process[j_id - 1]
                    self.activate_info(j_id, op_id, m_id, j_index, avail_job_pt, T)

    def check_sim_end_event(self,ind,pop):
        pop[ind].fitness.values = [500000]

    #-----------------------------------------------------------------------------------------------------------------
    def record_DS(self,avail_job,features_storage,T):
        random_bool = random.choice([True, False])
        if T > self.arrival_time[999] and len(avail_job) == 7 and len(self.DC) < 40 and random_bool:
            # 每一代都抽取40个决策场景，存features和gp个体计算出的rank
            self.DC.append(features_storage)
            # self.DC_rank.append(rank_c)

    def main_test(self,p,pop):
        e = True
        self.job_arrival_event(self.arrival_time)
        while not self.schedule_queue.empty():
            info = self.schedule_queue.get()
            T, T_type, task_no = info[0], info[1], info[2]
            # item,task_no = info[0],info[1]
            if T_type != 4:
                task_no = int("".join(list(filter(str.isdigit, task_no))))  # 从含字母的str提取数字
            # if not end:
            if T_type == 1:
                self.check_job_arrival(T, task_no)
            elif T_type == 2:
                self.check_machine_activate(T, task_no)
            elif T_type == 3:
                self.check_machine_work_done_event(T, task_no)
            elif T_type == 4:
                self.check_sim_end_event(p, pop)
                e = False
                break
                # self.check_sim_end_event(p,pop)
        if e:
            if (self.completion_time == 0).any():
                print("simulation not end")
                print(self.J_pool)
            else:
                self.get_flow_time()
            self.set_fitness(pop, self.Objective, self.flow_t, self.w_flow_t, p)

    def target_function(self):
        e = True
        self.job_arrival_event(self.arrival_time)
        while not self.schedule_queue.empty():
            info = self.schedule_queue.get()
            T, T_type, task_no = info[0], info[1], info[2]
            # item,task_no = info[0],info[1]
            if T_type != 4:
                task_no = int("".join(list(filter(str.isdigit, task_no))))  # 从含字母的str提取数字
            # if not end:
            if T_type == 1:
                self.check_job_arrival(T, task_no)
            elif T_type == 2:
                self.check_machine_activate(T, task_no)
            elif T_type == 3:
                self.check_machine_work_done_event(T, task_no)
            elif T_type == 4:
                e = False
                return 500000
        if e:
            if (self.completion_time == 0).any():
                print("simulation not end")
                print(self.J_pool)
            else:
                self.get_flow_time()
            fit = self.cal_fitness(self.Objective, self.flow_t, self.w_flow_t)
            return fit

    def target_sim(self,ind):
        self.Objective = "Mean-FlowTime"
        self.set_task(0.85,1,99)
        self.func = gp.compile(expr=ind, pset=self.pset)
        self.init_sim_param()
        self.init_features()
        self.schedule_queue = queue.PriorityQueue()
        fit = self.target_function()
        return fit

    def main_evolution(self):
        best_inds = []
        gen_min_fit = []
        best_rule_size = []
        test_gen_mean_fit = []
        way = ["w", "a"]
        data = []
        self.Seed = 510000
        self.Objective = "Mean-FlowTime"
        while self.gen < self.NGEN:
            self.set_task_two()
            for p in range(self.popSize):
                self.func = gp.compile(expr=self.pop[p], pset=self.pset)
                self.init_sim_param()
                self.init_features()
                self.schedule_queue = queue.PriorityQueue()
                self.main_test(p,self.pop)
                # print(self.pop[p].fitness.values[0])
                if self.pop[p].fitness.values[0] !=500000:
                    fit = statistics.mean(self.flow_t[self.warm_up_J:])
                    self.pop[p].fitness.values = [fit]
                    if fit != self.pop[p].fitness.values[0]:
                        print("unsame")
                if self.pop[p].fitness.values[0] < 0:
                    print("error")
                data.append(self.pop[p].fitness.values[0])
            res_pop = copy.deepcopy(self.pop)
            res_pop.sort(key=lambda x: x.fitness, reverse=True)
            best_inds.append(self.toolbox.clone(res_pop[0]))
            best_rule_size.append(len(res_pop[0]))
            gen_min_fit.append(min(data))
            if self.gen < self.NGEN - 1:
                elite_pop = [self.toolbox.clone(ind) for ind in res_pop[:self.elites + 1]]
                offspring = algorithms.varOr(population=self.pop, toolbox=self.toolbox, lambda_=self.popSize - self.elites, cxpb=self.cxPb,
                                             mutpb=self.mutPb)
                new_pop = offspring + elite_pop
                self.pop[:] = new_pop
            self.Seed += self.rotate
            self.gen += 1
        self.save_info("w", gen_min_fit)

    def generate_instance(self,max_pt):
        self.init_job_param()
        for i in range(self.num_J):
            self.num_ops[i], self.num_options[i], self.weights[i], self.m_of_op[i], self.pt_of_op[
                i] = self.generate_jobs_attribute_use_np(max_pt)
            if i > 0:
                self.arrival_time[i] = self.arrival_time[i - 1] + self.arrival_interval[i]
            # sum_mid_t, op_mid_pt = link_operations(num_ops, op_pt_time,i)  # mid_t作为初始mkr，更新是process = 2时
            self.link_operations_use_np(self.num_ops, i)

    def set_task_one(self):
        self.init_job_param_for_task()
        self.lamda = self.arrival_interval_lamda(1, 99, 0.75)
        self.arrival_interval = np.random.exponential(self.lamda, size=self.num_J)
        self.arrival_time[0] = self.arrival_interval[0]
        # self.init_job_param_for_task()
        # self.generate_instance(99)
        for i in range(self.num_J):
            self.num_ops[i], self.num_options[i], self.weights[i], self.m_of_op[i], self.pt_of_op[
                i] = self.generate_jobs_attribute_use_np(99)
            if i > 0:
                self.arrival_time[i] = self.arrival_time[i - 1] + self.arrival_interval[i]
            self.link_operations_use_np(self.num_ops, i)
            due_date = self.get_due_date(i)

    def set_task_two(self):
        self.init_job_param_for_task()
        self.lamda = self.arrival_interval_lamda(1, 49, 0.95)
        self.arrival_interval = np.random.exponential(self.lamda, size=self.num_J)
        self.arrival_time[0] = self.arrival_interval[0]
        # self.generate_instance(49)
        for i in range(self.num_J):
            self.num_ops[i], self.num_options[i], self.weights[i], self.m_of_op[i], self.pt_of_op[
                i] = self.generate_jobs_attribute_use_np(49)
            if i > 0:
                self.arrival_time[i] = self.arrival_time[i - 1] + self.arrival_interval[i]
            self.link_operations_use_np(self.num_ops, i)
            self.due_date = self.get_due_date(i)

    def set_task(self,util,minPT,maxPT):
        self.init_job_param_for_task()
        self.lamda = self.arrival_interval_lamda(minPT, maxPT, util)
        self.arrival_interval = np.random.exponential(self.lamda, size=self.num_J)
        self.arrival_time[0] = self.arrival_interval[0]
        # self.generate_instance(49)
        for i in range(self.num_J):
            self.num_ops[i], self.num_options[i], self.weights[i], self.m_of_op[i], self.pt_of_op[
                i] = self.generate_jobs_attribute_use_np(maxPT)
            if i > 0:
                self.arrival_time[i] = self.arrival_time[i - 1] + self.arrival_interval[i]
            self.link_operations_use_np(self.num_ops, i)
            self.due_date[i] = self.get_due_date(i)

    def full_tree(self,psize,transPop):
        res = self.target_pop + transPop
        self.target_pop[:] = res

    #选择最后一代前k％个体，取每个个体的随机子树
    def sub_tree(self,psize,transPop):
        for i in range(len(transPop)):
            ind = transPop[i]
            types1 = defaultdict(list)
            for idx, node in enumerate(ind[1:], 1):
                types1[node.ret].append(idx)
            tree1_types = set(types1.keys())
            type1_ = random.choice(list(tree1_types))
            index1 = random.choice(types1[type1_])
            slice1 = ind.searchSubtree(index1)
            n_ind = creator.Individual(gp.PrimitiveTree(ind[slice1]))
            self.target_pop.append(self.toolbox.clone(n_ind))

    def removeDuplicate(self, indList):
        l = len(indList)
        newInds = []
        indList.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(l):
            if indList[i].fitness.values[0] < math.inf:
                for j in range(i + 1, l):
                    delta = indList[j].fitness.values[0] - indList[i].fitness.values[0]
                    if indList[j].fitness.values[0] < math.inf and delta == 0:
                        indList[j].fitness.values = [math.inf]
                newInds.append(self.toolbox.clone(indList[i]))
        return newInds


    def save_info(self,way,*args):
        with open(self.file_name, way) as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow(args)

    def draw_plt(self,data):
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(15, 5))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("A test graph")
        x = [i for i in range(self.NGEN)]
        for i in range(len(data)):
            y = data[i]
            plt.plot(x, y, label='id %s' % i)
        plt.legend()
        plt.savefig(r'data/w_f.png', dpi=200)
        plt.show()

    def root_node_type(self,name:str):
        if name == 'max':
            return 0
        elif name == 'min':
            return 1
        elif name == 'add':
            return 2
        elif name == 'sub':
            return 3
        elif name == 'mul':
            return 4

    def node_type(self,name:str):
        if name == 'max':
            return 0
        elif name == 'min':
            return 1
        elif name == 'add':
            return 2
        elif name == 'sub':
            return 3
        elif name == 'mul':
            return 4
        elif name == 'protectedDiv':
            return 5
        elif name == 'ARG0':
            return 6
        elif name == 'ARG1':
            return 7
        elif name == 'ARG2':
            return 8
        elif name == 'ARG3':
            return 9
        elif name == 'ARG4':
            return 10
        elif name == 'ARG5':
            return 11
        elif name == 'ARG6':
            return 12
        elif name == 'ARG7':
            return 13
        elif name == 'ARG8':
            return 14
        elif name == 'ARG9':
            return 15
    #统计所有个体相同层级出现端点次数
    def level_probability(self):
        root_type = [[]for _ in range(6)]
        left_type = [[0 for _ in range(16)] for _ in range(5)]
        right_type = [[0 for _ in range(16)] for _ in range(5)]
        Node = namedtuple('node', 'id, name, p_id')
        p = self.toolbox.population(n=6)
        for i in range(len(p)):
            ind = p[i]
            node_arrangement = [[]for _ in range(2**ind.height - 1)]
            last_parentNode = 1
            types1 = defaultdict(list)
            for idx, node in enumerate(ind[0:], 1):
                types1[node.ret].append(idx)
                if idx == 1:
                    rt = self.node_type(node.name)
                    root_type[rt].append(i)
                    lid = 2 * idx
                    rid = 2 * idx + 1
                #二叉树中为k的父节点，它的左子节点下标为2k+1，右子节点是2k+2
                if idx == (2 * last_parentNode):
                    rt = self.node_type(ind[last_parentNode].name)
                    left_type_index = self.node_type(node.name)
                    left_type[rt][left_type_index] += 1
                # elif idx == (2 * i) + 2:
                #     left_type_index = self.node_type(node.name)
                #     if left_type_index > 4:
                #         left_type_index -= 5
                #     left_type[left_type_index] += 1
                last_parentNode += 1
        print(root_type)
        print(left_type)
            # tree1_types = set(types1.keys())
            # type1_ = random.choice(list(tree1_types))
            # index1 = random.choice(types1[type1_])
            # slice1 = ind.searchSubtree(index1)
            # n_ind = creator.Individual(gp.PrimitiveTree(ind[slice1]))

    def multitask_set(self):
        best_inds = []
        best_rule_size = []
        task_num = 2
        task_para = [0.75,(1,99),"Max-FlowTime",0.95,(1,49),"Mean-FlowTime"]
        self.pop[:] = self.toolbox.population(n= self.popSize)
        while self.gen < self.NGEN:
            fit_set = np.zeros(self.popSize)
            fit1 = np.zeros(self.popSize)
            fit2 = np.zeros(self.popSize)
            for t in range(task_num):
                if t == 0:
                # expr1 = "self.set_task(task_para{})"
                    self.set_task(0.75,1,99)
                    self.Objective = "Mean-FlowTime"
                else:
                    self.set_task(0.95, 1, 69)
                    self.Objective = "Mean-FlowTime"

                for p in range(self.popSize):
                    self.func = gp.compile(expr=self.pop[p], pset=self.pset)
                    self.init_sim_param()
                    self.init_features()
                    self.schedule_queue = queue.PriorityQueue()
                    self.main_test(p, self.pop)
                    # if fit < 0:
                    #     print("error1")
                    if self.pop[p].fitness.values[0] == 500000:
                        expr1 = "fit{}[p] = math.inf".format(t+1)
                        exec(expr1)
                        # fit1[p] = math.inf
                    else:
                        fit = self.cal_fitness(self.Objective, self.flow_t, self.w_flow_t)
                        expr2 = "fit{}[p] = fit".format(t + 1)
                        exec(expr2)
                        # fit1[p] = fit
            rank1 = rankdata(fit1,"min")
            rank2 = rankdata(fit2,"min")
            fit_set = np.add(fit1,fit2)
            fit = zip(fit1,fit2)
            rank3 = rankdata(fit_set,"min")
            rank4 = rankdata(np.add(rank1,rank2),'min')
            temp = sorted(enumerate(rank4), key=lambda x: rank4[1])  # x[1]是因为在enumerate(a)中，a数值在第1位
            rankIdx = [rank4[0] for rank4 in temp]  # 获取排序好后b坐标,下标在第0位
            fit_L = list(np.true_divide(fit_set, task_num))
            # fit_L = list(np.true_divide(fit_set,task_num))
            for ind in range(self.popSize):
                self.pop[ind].fitness.values = [rank4[ind]]
            res_pop = copy.deepcopy(self.pop)
            res_pop.sort(key = lambda ind: ind.fitness, reverse=True)
            # a = zip(rank4, self.pop)
            # tmp = sorted(list(zip(rank4, self.pop)))
            # result_list = [i for _, i in sorted(list(zip(rank4, self.pop)))]
            # print(self.pop[p].fitness.values[0])
            print("fit1",fit1)
            print("fit2",fit2)
            print("rank1", rank1)
            print("rank2", rank2)
            print("rank3", rank3)
            print("rank4", rank4)
            best_inds.append(self.toolbox.clone(res_pop[0]))
            best_rule_size.append(len(res_pop[0]))
            # gen_min_fit.append(min(data))
            if self.gen < self.NGEN - 1:
                elite_pop = [self.toolbox.clone(ind) for ind in res_pop[:self.elites + 1]]
                offspring = algorithms.varOr(population=self.pop, toolbox=self.toolbox,
                                             lambda_=self.popSize - self.elites, cxpb=self.cxPb,
                                             mutpb=self.mutPb)
                new_pop = offspring + elite_pop
                self.pop[:] = new_pop
            self.Seed += self.rotate
            self.gen += 1
        return best_inds

    def tree_trans(self):
        ind = self.pop[0]
        code = str(ind)
        tokens = re.split("[ \t\n\r\f\v(),]", code)
        if len(self.pset.arguments) > 0:
            # This section is a stripped version of the lambdify
            # function of SymPy 0.6.6.
            args = ",".join(arg for arg in self.pset.arguments)
            code = "lambda {args}: {code}".format(args=args, code=code)
            a = list(filter(lambda x: x in self.pset.arguments, tokens))
            a = list(set(a))
        # val = eval(code, self.pset.context, {})
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

    def count_param(self,ind):
        # ind = self.pop[0]
        code = str(ind)
        tokens = re.split("[ \t\n\r\f\v(),]", code)
        if len(self.pset.arguments) > 0:
            # This section is a stripped version of the lambdify
            # function of SymPy 0.6.6.
            # arg = ",".join(arg for arg in self.pset.arguments)
            terminals = self.pset.arguments
            args = [x for x in tokens if x in terminals]
            # c = list(set(b) & set(tokens))
            # arg = [lambda x: x in self.pset.arguments and x != '', tokens]
            # code = "lambda {args}: {code}".format(args=arg, code=code)
            # args = filter(lambda x: x in self.pset.arguments, tokens)
            # a = eval(code, self.pset.context, {})
            # fil_args = list(set(args))
        return args

    def random_ind(self):
        p = self.toolbox.population(n= 1)
        return p[0]

    def compile_tree(self,expr,pset,argl):
        """Compile the expression *expr*.

        :param expr: Expression to compile. It can either be a PrimitiveTree,
                     a string of Python code or any object that when
                     converted into string produced a valid Python code
                     expression.
        :param pset: Primitive set against which the expression is compile.
        :returns: a function if the primitive set has 1 or more arguments,
                  or return the results produced by evaluating the tree.
        """
        code = str(expr)
        if len(pset.arguments) > 0:
            # This section is a stripped version of the lambdify
            # function of SymPy 0.6.6.
            args = ",".join(arg for arg in argl)
            code = "lambda {args}: {code}".format(args=args, code=code)
        # try:
            return eval(code, pset.context, {})
        # except MemoryError:
        #     _, _, traceback = sys.exc_info()
        #     raise MemoryError("DEAP : Error in tree evaluation :"
        #                       " Python cannot evaluate a tree higher than 90. "
        #                       "To avoid this problem, you should use bloat control on your "
        #                       "operators. See the DEAP documentation for more information. "
        #                       "DEAP will now abort.").with_traceback(traceback)




