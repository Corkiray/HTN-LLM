#!/usr/bin/env python
# coding: utf-8

import os

#################### Execute the LLM Planner over a plans set whose structure is: plan_path/domain_name/problem+'_plan'#####################3
ROOT_PATH = '/mnt/homeGPU/ipuerta/HTN-LLM/'
MODEL_PATH ='/mnt/homeGPU/ipuerta/Models/neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16'

DATASET_PATH = os.path.join(ROOT_PATH, 'ipc2023-domains-main/total-order')
VALIDATOR_PATH = os.path.join(ROOT_PATH, 'pandaPIparser-master/pandaPIparser')
PLANS_PATH = os.path.join(ROOT_PATH, 'ipc2023-generated-plans_Original')
RESULTS_PATH = os.path.join(ROOT_PATH, 'ipc2023-validator-results')
DOMAINS_LIST = os.listdir(DATASET_PATH)
print(DOMAINS_LIST)

total_words = 0

# iterate over domains
for domain_name in DOMAINS_LIST:
    words = 0    
    domain_problems_path = os.path.join(DATASET_PATH, domain_name)
    specific_domain_path = os.path.join(domain_problems_path, 'domain.hddl')
    if 'Monroe' not in specific_domain_path:
        with open(specific_domain_path) as f:
            domain_words = len(f.read().split())

    #iterate over problems
    for problem_name in os.listdir(domain_problems_path):   
        if problem_name != 'domain.hddl':
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
                problem_words = len(f.read().split())

            domain_plans_path = os.path.join(PLANS_PATH, domain_name)
            plan_path = os.path.join(domain_plans_path, problem_name + '_plan')
            
            if os.path.isfile(plan_path):
                with open(plan_path) as f: 
                    plan_words = len(f.read().split())
                words = domain_words + problem_words + plan_words          

    print(words, "words in", domain_name, "planning")
    total_words += words

print(total_words, "total words")