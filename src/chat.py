from openai import OpenAI

def chat_with_gpt():
    MODEL_NAME = "01-ai/Yi-Coder-9B-Chat"
    openai_completions_api_version = "v1"
    beam_api_url = "https://yicoder-inference-server-259d00e-v1.app.beam.cloud"

    client = OpenAI(
        api_key="",
        base_url=f"{beam_api_url}/{openai_completions_api_version}",
    )

    conversation_history = []

    print("Chat Starting...")

    if client.models.list().data[0].id == MODEL_NAME:
        print("Model is ready")
    else:
        print("Failed to load model")
        exit(1)

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            conversation_history.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=conversation_history
            )
            assistant_reply = response.choices[0].message.content
            conversation_history.append(
                {"role": "assistant", "content": assistant_reply}
            )

            print("Assistant:", assistant_reply)

    except KeyboardInterrupt:
        print("\nExiting the chat.")


if __name__ == "__main__":
    chat_with_gpt()