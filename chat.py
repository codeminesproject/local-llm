import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answer. You answer the question in short and consise way"
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()
    question = get_prompt(message.content, message_history)
    response = ""
    for word in llm(question, stream=True):
        await msg.stream_token(word)
        response += word
    message_history.append(response)
    await msg.update()


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )


"""
history = []

question = "what is capital of india?"

answer = ""

result = llm(get_prompt(question))
print(result)
answer += result

history.append(answer)

print("=========================================")
print(history)
print("==========================================")
question = "and which is of the united states?"

result = llm(get_prompt(question, history))
print(result)
"""
