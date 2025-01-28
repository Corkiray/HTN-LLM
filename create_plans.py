#!/usr/bin/env python
# coding: utf-8

import os, sys, torch, time
from LLM_Loaders import OpenRouterAPI as llm_loader
from LLM_Planners import Simple_LLMPlanner as llm_planner

#################### Execute the LLM Planner over a plans set whose structure is: plan_path/domain_name/problem+'_plan'#####################3
ROOT_PATH = '/mnt/homeGPU/ipuerta/HTN-LLM/'
MODEL_PATH ='/mnt/homeGPU/ipuerta/Models/neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16'

DATASET_PATH = os.path.join(ROOT_PATH, 'ipc2023-domains-main/total-order')
VALIDATOR_PATH = os.path.join(ROOT_PATH, 'pandaPIparser-master/pandaPIparser')
PLANS_PATH = os.path.join(ROOT_PATH, 'ipc2023-generated-plans')
RESULTS_PATH = os.path.join(ROOT_PATH, 'ipc2023-validator-results')

DOMAINS_LIST = ['AssemblyHierarchical', 'Barman-BDI', 'Blocksworld-HPDDL', 'Depots', 'Factories-simple',
'Freecell-Learned-ECAI-16', 'Hiking', 'Lamps', 'Logistics-Learned-ECAI-16', 'Multiarm-Blocksworld',
'Robot', 'Satellite-GTOHP', 'Towers', 'Transport', 'Woodworking']

# --------------------Loading the Model------------------------------------
llm = llm_loader()

# --------------------Instantiating the Planner------------------------------------
planner = llm_planner(llm)

# ----------------------- Generating the plans -----------------------------
# Iteratively calls the model with the dataset domains and plans

#iterate over domains
for domain_name in DOMAINS_LIST:
    print('Planning for the domain ', domain_name)
    domain_problems_path = os.path.join(DATASET_PATH, domain_name)
    specific_domain_path = os.path.join(domain_problems_path, 'domain.hddl')
    problems_list = sorted(os.listdir(domain_problems_path))

    #TODO THIS IS TEMPORAL, ERASE IT
    problems_list = [name[:-5] for name in sorted(os.listdir(os.path.join(
        '/mnt/homeGPU/ipuerta/HTN-LLM/ipc2023-generated-plans_Original', domain_name)))]
        
    if 'Monroe' not in specific_domain_path:
        with open(specific_domain_path) as f:
            domain_info = f.read()

    #iterate over problems
    for problem_name in problems_list:       
        if problem_name != 'domain.hddl':
            print('Planning for', problem_name)
            time.sleep(0.1)
            specific_problem_path = os.path.join(domain_problems_path, problem_name)

            # Monroe's exception, which has an specific domain file for each problem
            if 'Monroe' in domain_name:
                if 'domain' in problem_name:
                    specific_domain_path = os.path.join(domain_problems_path, problem_name)
                    with open(specific_domain_path) as f:
                        domain_info = f.read()
                    specific_problem_path = specific_domain_path.replace('-domain.hddl', '.hddl')
                else: continue

            with open(specific_problem_path) as f:
                problem_info = f.read()

            #-----Obtaining and saving the plan-----
            domain_plans_path = os.path.join(PLANS_PATH, domain_name)
            plan_path = os.path.join(domain_plans_path, problem_name + '_plan')
            
            #If doesn't exists the folder to allocate the specific domain plans (i.e. first time generating for this domain), it is created
            if not os.path.exists(domain_plans_path):
                os.makedirs( os.path.join(PLANS_PATH, domain_name))
            
            #If already exists the plan (i.e. obtained in another execution), pass
            if os.path.exists(plan_path): 
                print('\tPlan already generated')
            else:
                # To every pair (domain_info, problem_info), generate the plan
                plan = planner.make_plan(domain_info, problem_info)
                # torch.cuda.memory._dump_snapshot(file+'_snapshot.pickle')
                with open(plan_path, 'w+') as f: 
                    f.write(plan)            
