"""Plain-text Base fallback, explicitly separate from trained tool/code protocols."""
import time

def general_prefix(prompt):
    return ('Question: What is 1 + 1?\nAnswer: 2\n\n'
            'Question: What color is a clear daytime sky?\nAnswer: blue\n\n'
            'Question: '+prompt.strip()+'\nAnswer:')

def run_general_generation(model,tokenizer,torch,prompt,max_new_tokens):
    from transformers import StoppingCriteria,StoppingCriteriaList
    inputs=tokenizer(general_prefix(prompt),return_tensors='pt',add_special_tokens=False).to('cuda')
    length=inputs['input_ids'].shape[1]
    class StopAnswer(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=tokenizer.decode(input_ids[0,length:],skip_special_tokens=False)
            return '\nQuestion:' in text or '<|im_end|>' in text
    torch.cuda.synchronize(); start=time.monotonic()
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
            pad_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
            stopping_criteria=StoppingCriteriaList([StopAnswer()]))
    torch.cuda.synchronize(); elapsed=time.monotonic()-start
    tokens=output[0,length:]; raw=tokenizer.decode(tokens,skip_special_tokens=False)
    answer=raw.split('\nQuestion:',1)[0].split('<|endoftext|>',1)[0].split('<|im_end|>',1)[0].strip()
    return {'output_text':raw,'answer':answer,'generated_tokens':len(tokens),
            'latency_seconds':elapsed,'hit_token_limit':len(tokens)==max_new_tokens}
