from beam import Image, Volume, env, asgi

if env.is_remote():
    import asyncio
    import fastapi
    import vllm.entrypoints.openai.api_server as openai_api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.usage.usage_lib import UsageContext


MODEL_NAME = "01-ai/Yi-Coder-9B-Chat"

vllm_cache = Volume(name="yicoder", mount_path="./yicoder")

@asgi(
    image=Image().add_python_packages(["vllm"]),
    volumes=[vllm_cache],
    gpu="A100-40",
    memory="8Gi",
    cpu=1,
    keep_warm_seconds=360,
)
def yicoder_api():
    app = fastapi.FastAPI(
        title=f"{MODEL_NAME} API",
        docs_url="/docs",
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    app.include_router(openai_api_server.router)

    # create engine client
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=8096,
        download_dir=vllm_cache.mount_path,
    )
    async_engine_client = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    model_config = asyncio.run(async_engine_client.get_model_config())

    # request logger
    request_logger = RequestLogger(max_log_len=2048)

    # setup open-ai serving chat and completion endpoints
    openai_api_server.openai_serving_chat = OpenAIServingChat(
        async_engine_client,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        chat_template=None,
        lora_modules=[],
        prompt_adapters=[],
        response_role="assistant",
        request_logger=request_logger,
    )

    openai_api_server.openai_serving_completion = OpenAIServingCompletion(
        async_engine_client,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return app
