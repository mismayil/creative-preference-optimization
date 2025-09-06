import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# get dtype
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16


class EndpointHandler:
    def __init__(self, path=""):
        # load the model
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=dtype)
        # create inference pipeline
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, list):
            if all(isinstance(i, str) for i in inputs):
                inputs = [[{"role": "user", "content": i}] for i in inputs]
            elif all(isinstance(i, dict) for i in inputs):
                # assume the list is already in the correct format
                pass
            else:
                raise ValueError("Inputs must be a string or a list of strings or dictionaries.")
        # pass inputs with all kwargs in data
        if parameters is not None:
            predictions = self.pipeline(inputs, **parameters)
        else:
            predictions = self.pipeline(inputs)
        # postprocess the prediction
        responses = [{"prompt": prediction[0]["generated_text"][0]["content"], "response": prediction[0]["generated_text"][1]["content"]} for prediction in predictions]
        return responses

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage
    handler = EndpointHandler(path="CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-cre")
    input_data = {
        "inputs": ["Come up with an original and creative use for the following object: rope", 
                   "Finish the sentence with an original and creative ending: When I got on the school bus...."], 
        "parameters": {"top_p": 0.95, "temperature": 0.7, "max_new_tokens": 256, "do_sample": True}
    }
    
    # Simulate receiving data from a request
    response = handler(input_data)
    
    # Print the response
    print(json.dumps(response, indent=2))