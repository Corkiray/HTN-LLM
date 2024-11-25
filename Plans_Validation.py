#!/usr/bin/env python
# coding: utf-8

################### Executes the validator over a plans set whose structure is: plan_path/domain_name/problem+'_plan'##############################
ROOT_PATH = '/mnt/homeGPU/ipuerta/HTN-LLM/'

import os, subprocess, re

DATASET_PATH = os.path.join(ROOT_PATH, 'ipc2023-domains-main/total-order')
VALIDATOR_PATH = os.path.join(ROOT_PATH, 'pandaPIparser-master/pandaPIparser')
PLANS_PATH = os.path.join(ROOT_PATH, 'ipc2023-generated-plans')
RESULTS_PATH = os.path.join(ROOT_PATH, 'ipc2023-validator-results')

DOMAINS_LIST = os.listdir(PLANS_PATH)
print(DOMAINS_LIST)

# iterate over domains
for domain_name in DOMAINS_LIST:
    print('Verifying the domain ', domain_name)
    domain_problems_path = os.path.join(DATASET_PATH, domain_name)
    domain_plans_path = os.path.join(PLANS_PATH, domain_name)
    specific_domain_path = os.path.join(domain_problems_path, 'domain.hddl')

    num_problems = len(os.listdir(domain_plans_path))
    num_correct_plan = 0
    num_correct_plan_format = 0 
    num_correct_descomposition_format = 0
    num_correct_descomposition = 0

    #iterate over plans/problems
    for plan_name in os.listdir(domain_plans_path):
        problem_name = plan_name[:-5]
        print('\tProblem:', problem_name)
        specific_problem_path = os.path.join(domain_problems_path, problem_name)
        specific_plan_path= os.path.join(PLANS_PATH, domain_name, plan_name)

        output = subprocess.run(
            args = [VALIDATOR_PATH, '--verify', '-C', specific_domain_path, specific_problem_path, specific_plan_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')
        
        if re.search('IDs of subtasks used in the plan exist: true', output):
            num_correct_plan_format+=1
        if re.search('Primitive plan alone executable: true', output):
            num_correct_plan+=1
        if re.search('Tasks declared in plan actually exist and can be instantiated as given: true', output):
            num_correct_descomposition_format+=1
        if re.search('Plan verification result: true', output):
            num_correct_descomposition+=1

    with open(os.path.join(RESULTS_PATH, domain_name + '_results'), 'w+') as f: 
        f.write(f'''Total_Problems {num_problems}
Correct_Plan_Format {num_correct_plan_format}
Correct_Plan   {num_correct_plan}
Correct_Descomposition_Format   {num_correct_descomposition_format}
Correct_Descomposotion  {num_correct_descomposition}
''')