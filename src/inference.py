from beam import Image, Volume, env, function

# These imports are only available in the remote environment
if env.is_remote():
    from vllm import LLM

# This beam volume is mounted as a file system and used to cache the downloaded model
vllm_cache = Volume(name="yicoder", mount_path="./yicoder")

@function(
    image=Image().add_python_packages(["vllm"]),
    volumes=[vllm_cache],
    gpu="A100-40",
    memory="8Gi",
    cpu=1,
)
def yicoder(prompt: str):
    llm = LLM(
        model="01-ai/Yi-Coder-9B-Chat",
        download_dir=vllm_cache.mount_path,
        max_model_len=8096,
    )
    request_output = llm.chat(
        messages=[{"role": "user", "content": prompt}],
    )
    return request_output[0].outputs[0].text


if __name__ == "__main__":
    print(yicoder.remote("How can I use `echo` to say hi in my terminal?"))
