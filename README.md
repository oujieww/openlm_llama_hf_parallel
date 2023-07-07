# ParaLLaMA
# parallelize openlm llama for huggingface

we change the original code of modeling_llama.py from "huggingface", as we want to finetune the llama on limited resources
, as one 32G V100 and one 16G v100.

we need to use pipeline parallelize, init layers on different devices by "device_map".

if you want use our code, you need to replace the file(modeling_llama.py) in your python env.

```python
model = LlamaForCausalLM.from_pretrained(init_model_path)
device_map = {
    0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # in my machine, gpu 0 has 32G memory
    1: [19,20,21,22,23,24,25], #  gpu 1 has 16G memory
}
model.parallelize(device_map)
```

