import os, re

ORIGINAL_PLANS_PATH = "/mnt/homeGPU/ipuerta/ipc-2020-plans/plans/IPC-2020"
NEW_PLANS_PATH = "/mnt/homeGPU/ipuerta/ipc2020-plans-reformatted"

def reformat(plan):
    new_plan = []
    for number, action in enumerate(plan, start=1):
        action_name, action_params = re.search('(.+?)\\[(.*?)\\]', action).group(1,2)
        action_params = action_params.split(',')
        new_plan.append(str(number) + " " + action_name + " " + " ".join(action_params))
    return "\n".join(new_plan)

for original_name in os.listdir(ORIGINAL_PLANS_PATH):
    with open(os.path.join(ORIGINAL_PLANS_PATH, original_name)) as f:
        domain_path = f.readline().split('/')
        problem_path = f.readline().split('/')
        original_plan = f.readline().split(';')
    
    if problem_path[-2] != domain_path[-2]:
        print('ERROR: Discrepancy between the problem and domain paths')

    new_name = domain_path[-2] + '__' + domain_path[-1][:-1] + '__' + problem_path[-1][:-1]
    with open(os.path.join(NEW_PLANS_PATH, new_name), 'w+') as f: 
        f.write(reformat(original_plan))