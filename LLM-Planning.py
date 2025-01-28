#!/usr/bin/env python
# coding: utf-8

import os, sys, torch
from LLM_Loaders import General_LLM

#################### Execute the LLM Planner over a plans set whose structure is: plan_path/domain_name/problem+'_plan'#####################3
ROOT_PATH = '/mnt/homeGPU/ipuerta/HTN-LLM/'
MODEL_PATH ='/mnt/homeGPU/ipuerta/Models/2Llama-3.1-Nemotron-70B-Instruct-HF'

DATASET_PATH = os.path.join(ROOT_PATH, 'ipc2023-domains-main/total-order')
VALIDATOR_PATH = os.path.join(ROOT_PATH, 'pandaPIparser-master/pandaPIparser')
PLANS_PATH = os.path.join(ROOT_PATH, 'ipc2023-generated-plans')
RESULTS_PATH = os.path.join(ROOT_PATH, 'ipc2023-validator-results')

DOMAINS_LIST = os.listdir(PLANS_PATH)
print(DOMAINS_LIST)

#Prompting framework utilized for the LLM Planner execution
def Make_LLM_Plan(llm: General_LLM, domain_info, problem_info):
    context= f'''You have to give an hierarquical plan, which consists of a sequential plan followed by the hierarchical descompositions. Use two arrows to indicate when the plan starts an ends, and the word root to indicate the start of the descomposition.
Here is an example of an output format:
==>
3 move f0 f1
4 board p0 f1
5 move f1 f0
6 debark p0 f0
root 0
0 solve_elevator -> m1_go_ordering_0 1 2
1 deliver_person p0 f1 f0 -> m2_ordering_0 3 4 5 6
2 solve_elevator -> m1_abort_ordering_0
<=='''

    prompt = f'''Given the next HTN Domain:
    {domain_info}
    And the next HTN Problem:
    {problem_info}
    Return a plan to solve it.
    '''
    messages = [{"role": "system", "content": context},
                {"role": "user", "content": prompt}]

    return llm.call(messages)


# --------------------Loading the Model------------------------------------

#Making sure that the python environment is correct
# print(sys.executable)

#Making sure all GPUs are availeble
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# if torch.cuda.is_available():
#     print('__CUDNN VERSION:', torch.backends.cudnn.version())
#     print('__Number CUDA Devices:', torch.cuda.device_count())
#     # torch.cuda.set_per_process_memory_fraction(0.7)
#     for device in range(torch.cuda.device_count()):
#         print(f'__CUDA Device {device}: \tName: {torch.cuda.get_device_name(device)}\t'
#                 +f'Memory [GB]:{torch.cuda.get_device_properties(device).total_memory/1e9}')
#     # torch.cuda.memory._record_memory_history()
# else: print('ERROR: cuda not avalible')

#Charging the model into a HF Pipeline
# config = LlamaConfig.from_json_file(os.path.join(MODEL_PATH, "config.json"))
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer.pad_token = tokenizer.unk_token
# tokenizer.pad_token_id = tokenizer.unk_token_id
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", 
#                             max_memory={0: "18GiB", 1:"18GiB", 2:"18GiB", 3:"18GiB", 4:"18GiB", 5:"18GiB", "cpu":"360GiB"}, 
#                             config=config, offload_state_dict=True)
# model = pipeline('text-generation', model=model, tokenizer=tokenizer)
# torch.cuda.memory._dump_snapshot('ModelLoad_snapshot.pickle')
# print(f"Model device: {model.device}")
# print(f'Mapa del modelo:\n {model.device_map}')

llm = General_LLM(MODEL_PATH)

# ----------------------- Generating the plans -----------------------------
# Iteratively calls the model with the dataset domains and plans

# iterate over domains
for domain_name in DOMAINS_LIST:
    print('Planning for the domain ', domain_name)
    domain_problems_path = os.path.join(DATASET_PATH, domain_name)
    specific_domain_path = os.path.join(domain_problems_path, 'domain.hddl')
    with open(specific_domain_path) as f:
        domain_info = f.read()

    #iterate over problems
    for problem_name in os.listdir(domain_problems_path):       
        if problem_name != 'domain.hddl':
            print('Planning for', problem_name)
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

            # To every pair (domain_info, problem_info), generate the plan
            plan = Make_LLM_Plan(llm=llm, domain_info=domain_info, problem_info=problem_info)
            # torch.cuda.memory._dump_snapshot(file+'_snapshot.pickle')

            domain_plans_path = os.path.join(PLANS_PATH, domain_name)
            
            #If doesn't exists the folder to allocate the specific domain plans (i.e. first time generating for this domain), it is created
            if not os.path.exists(domain_plans_path):
                os.makedirs( os.path.join(PLANS_PATH, domain_name))

            plan_path = os.path.join(domain_plans_path, problem_name + '_plan')
            with open(plan_path, 'w+') as f: 
                f.write(plan)
            
