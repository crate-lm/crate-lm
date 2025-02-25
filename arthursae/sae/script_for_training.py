import subprocess
import os
import multiprocessing

def run_script(keywords, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = ["python", "./main.py"]
    for key, value in keywords.items():
        args.append(f"--{key}={value}")
    print(f"Running on GPU {gpu_id}: " + " ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':
    jobs = []
    keyword_list = []

    num_gpus = 8  # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    model_class = "crate"
    # model_class = "gpt"

    # Example configurations
    model_name_list = [f'{model_class}-1l', f'{model_class}-2l', f'{model_class}-3l']
    layer_list = [[0], [0, 1], [0, 1, 2]]
    d_sae_list = [[128 * 4 * 16], [128 * 4 * 16, 128 * 4 * 16], [128 * 4 * 16, 128 * 4 * 16, 128 * 4 * 16]]
    d_in_list = [[128 * 4], [128 * 4, 128 * 4], [128 * 4, 128 * 4, 128 * 4]]

    for i in range(len(model_name_list)):
        model_name = model_name_list[i]
        layers = layer_list[i]
        d_saes = d_sae_list[i]
        d_ins = d_in_list[i]
        for j in range(len(layers)):
            layer = layers[j]
            d_sae = d_saes[j]
            d_in = d_ins[j]
            keyword_list.append({
                "model_name": model_name,
                "layer": layer,
                "act_name": f"blocks.{layer}.mlp.hook_post",
                "d_sae": d_sae,
                "d_in": d_in,
                "dataset_split": "test",
                "delete_cache": True,
                "l1_lambda": 8e-5,
                "lr": 4e-4,
            })

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)

    assert len(keyword_list) <= num_gpus * num_jobs_per_gpu, "Too many jobs for the number of GPUs available"
    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = threshold_idx % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id)))
    
    for job in jobs:
        job.get()
