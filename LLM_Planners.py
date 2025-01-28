#!/usr/bin/env python
# coding: utf-8

import re, os, random

DATASET_PATH = '/mnt/homeGPU/ipuerta/HTN-LLM/ipc2020-domains-master/total-order'
CORPUS_PATH = '/mnt/homeGPU/ipuerta/ipc2020-plans-reformatted'

class Simple_LLMPlanner:
    context= f'''You have to give an hierarquical plan, which consists of a sequential plan followed by the hierarchical descompositions. Use two arrows to indicate when the plan starts an ends, and the word root to indicate the start of the descomposition.
Here is an example of an output format:
==>
1 move f0 f1
2 board p0 f1
3 move f1 f0
4 debark p0 f0
root 0
0 solve_elevator -> m1_go_ordering_0 1 2
5 deliver_person p0 f1 f0 -> m2_ordering_0 3 4 5 6
6 solve_elevator -> m1_abort_ordering_0
<=='''
    
    def __init__(self, llm):
        self.llm = llm

    def make_plan(self, domain_info, problem_info):
        prompt = f'''Given the next HTN Domain:
        {domain_info}
        And the next HTN Problem:
        {problem_info}
        Return a plan to solve it.
        '''
        messages = [{"role": "system", "content": self.context},
                    {"role": "user", "content": prompt}]

        output = self.llm.call(messages)
        print("output len =", len(output.split()))
        return output
    

class Basic1ShotCoT_LLMPlanner(Simple_LLMPlanner):
        context= f'''You have to give an hierarquical plan, which consists of an hierarchical descomposition and a sequential plan. 
Let's think step by step, first making the descomposition and finally showing the concluded sequential plan.
Here is an example of an output format:
- Hieraquical Descomosition[root 0
0 solve_elevator -> m1_go_ordering_0 1 2
1 deliver_person p0 f1 f0 -> m2_ordering_0 3 4 5 6
2 solve_elevator -> m1_abort_ordering_0]
- Sequential Plan[3 move f0 f1
4 board p0 f1
5 move f1 f0
6 debark p0 f0]
You have to strictly respect the format.
''' 
        def make_plan(self, domain_info, problem_info):
            llm_response = super().make_plan(domain_info, problem_info)
            descomposition = re.search('Hieraquical Descomosition\\[(.+?)\\]', llm_response).group(1)
            plan = re.search('Sequential Plan\\[(.+?)\\]', llm_response).group(1)
            return '==>'+plan+'\n'+descomposition+'<=='
        

class BasicPromptLearning_LLMPlanner(Simple_LLMPlanner):
    learning_type = "out_of_context"
    num_examples = 5
    learning_prompt = "Here are some examples of generated automated planning sequential plans based on a hierarquical planning (HTN) domain and problem:\n"
    corpus_path = CORPUS_PATH
    dataset_path = DATASET_PATH

    def __init__(self, llm, corpus_path=corpus_path, num_examples=num_examples, learning_type=learning_type):
        if corpus_path != None: self.corpus_path = corpus_path
        if num_examples != None: self.num_examples = num_examples
        if learning_type != None: self.learning_type = learning_type

        self.corpus_list = os.listdir(corpus_path)
        super().__init__(llm)

    
    def make_plan(self, domain_info, problem_info, domain_name, problem_name):
        if self.learning_type == "out_of_context": examples_list = self.take_OutOfContext_examples(domain_name, problem_name)
        elif self.learning_type == "in_context": examples_list = self.take_InContext_examples(domain_name, problem_name)
        else: print("ERROR: UNKNOWN LEARNING TYPE") 

        examples = self.generate_examples_prompt(examples_list)

        self.context = self.learning_prompt + examples +'\n'+self.context
        return super().make_plan(domain_info, problem_info)
    
    def take_OutOfContext_examples(self, domain_name, problem_name):
        examples = [example_name for example_name in self.corpus_list 
                    if not domain_name in example_name and not problem_name in example_name]
        return random.sample(examples, self.num_examples)

    def take_InContext_examples(self, domain_name, problem_name):
        examples = [example_name for example_name in self.corpus_list 
                    if domain_name in example_name and not problem_name in example_name]
        return random.sample(examples, self.num_examples)

    def generate_examples_prompt(self, examples_list):
        output = ""
        for example_name in examples_list:
            example_domain_name, example_domain_file, example_problem_file = example_name.split('__')
            with open(os.path.join(self.dataset_path, example_domain_name, example_domain_file)) as f:
                example_domain_info = f.read()
            with open(os.path.join(self.dataset_path, example_domain_name, example_problem_file)) as f:
                example_problem_info = f.read()
            with open(os.path.join(self.corpus_path, example_name)) as f:
                example_plan_info = f.read()
                
            output += "HTN Domain:\n"+example_domain_info+"\nHTN Problem:\n"+example_problem_info+"\nSequential Plan["+example_plan_info+']\n\n'

        return output
    

class BasicPromptLearning_1ShotCoT_LLMPlanner(BasicPromptLearning_LLMPlanner):
    context = Basic1ShotCoT_LLMPlanner.context

    def make_plan(self, domain_info, problem_info, domain_name, problem_name):
        llm_response = super().make_plan(domain_info, problem_info, domain_name, problem_name)
        descomposition = re.findall('Hieraquical Descomosition\[(.+?)\]', llm_response)[-1]
        plan = re.findall('Sequential Plan\[(.+?)\]', llm_response)[-1]
        return '==>'+plan+'\n'+descomposition+'<=='

   