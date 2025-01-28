import sys
import json as js
import time
# from multiprocessing import freeze_support

KEYS_FILE = '/mnt/homeGPU/ipuerta/HTN-LLM/keys.json'

with open(KEYS_FILE) as f:
    keys_data = js.load(f)

#################### Execute the LLM Planner over a plans set whose structure is: plan_path/domain_name/problem+'_plan'#####################3
class StoredLLM:
    lib = "vllm"
    number_gpus = 2
    max_model_len = 8192
    max_tokens = 512

    def __ini__(self, model_path):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        
        if self.lib == "vllm":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_path, tensor_parallel_size=self.number_gpus, max_model_len=self.max_model_len)
            self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=self.max_tokens)
        elif self.lib=='transformers':
            from transformers import AutoModelForCausalLM, pipeline
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_state_dict=True)  
            self.llm = pipeline('text-generation', model=model, tokenizer=self.tokenizer)
        else:
            print("INIT ERROR: Lib not found")


    def call(self, messages):
        if self.lib =='vllm':
            prompts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            outputs = self.llm.generate(prompts, self.sampling_params)
            return outputs[0].outputs[0].text
        elif self.lib=='transformers':
            outputs = self.llm(messages)
            return outputs[0]["generated_text"][-1]['content']
        else:
            print("ERROR: Lib not found")

class GeminiAPI:
    default_version = "gemini-1.5-pro"
    max_requests_per_minute = 2
    max_tokens_per_minute = 32000

    def __init__(self, version = default_version):
        import google.generativeai as genai
        genai.configure(api_key=keys_data["GeminiApi"])
        self.model = genai.GenerativeModel(version)
        self.version = version
        self.requests_list=[]
    
    def call(self, messages):
        prompt = " ".join([item['content'] for item in messages])

        #Decide if the call is available based on the Gemini API's limitations. 
        # if it is not, wait until being available
        self.manage_availability(prompt)
        
        return self.model.generate_content(prompt).text
        
    def manage_availability(self, prompt):
        now = time.time()
        wait = True
        while wait:
            if len(self.requests_list) >= GeminiAPI.max_requests_per_minute:
                self.wait_until_forget_one_request(now)
            
            avalible_tokens = GeminiAPI.max_tokens_per_minute - sum([request['tokens'] for request in self.requests_list])
            necesary_tokens = self.model.count_tokens(prompt).total_tokens
            
            print('\tinput tokens =', necesary_tokens, end='\t')
            
            if avalible_tokens < necesary_tokens:
                self.wait_until_forget_one_request(now)
                continue
            
            wait = False

        self.requests_list.append({'time': time.time(), 'tokens': necesary_tokens})

    def wait_until_forget_one_request(self, now):
        print('\tWaiting for space on the list', self.requests_list)
        elapsed_time = now - self.requests_list[0]['time']
        if elapsed_time < 60:
            time.sleep(60-elapsed_time)
        self.requests_list.pop(0)

class OpenRouterAPI:
    def __init__(self):
        self.key = keys_data['OpenRouterAPI']

    def call(self, messages):
        import requests, json
        response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={    "Authorization": "Bearer "+self.key  },
                    data=json.dumps({
                        "model": "meta-llama/llama-3.1-70b-instruct:free", # Optional
                        "messages": messages
                        })
                    )
        print('\taprox input tokens =', len(str(messages)), end='\t')

        try:
            output = response.json()['choices'][0]['message']['content']
        except:
            print('ERROR: Couldnt handel with', response.json())
            time.sleep(5)
            output = self.call(messages)    

        return output
if __name__ == '__main__':
    # freeze_support()

    #Making sure that the environment is correct
    print(sys.executable)

    #Making sure all GPUs are availeble
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # if torch.cuda.is_available():
    #     print('__CUDNN VERSION:', torch.backends.cudnn.version())
    #     print('__Number CUDA Devices:', torch.cuda.device_count())
    #     # torch.cuda.set_per_process_memory_fraction(0.7)
    #     for device in range(torch.cuda.device_count()):
    #         print(f'__CUDA Device {device}: \tName: {torch.cuda.get_device_name(device)}\t'
    #                 +f'Memory [GB]:{torch.cuda.get_device_properties(device).total_memory/1e9}')
    #     # torch.cuda.memory._record_memory_history()
    # else: print('ERROR: cuda not avalible')
    # Llama_3_1_70B_Quantizied.number_gpus = torch.cuda.device_count()
    # print('Number of GPUs:', Llama_3_1_70B_Quantizied.number_gpus)
    
    # --------------------Loading the Model------------------------------------
    MODEL_PATH ='/mnt/homeGPU/ipuerta/Models/neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16'
    llm = OpenRouterAPI()

    #Calling the model
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    print(llm.call(messages))