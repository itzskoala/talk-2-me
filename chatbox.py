#So the first thing to do is call dependencies
#system prompt, user prompt, gradio ai for the UI, which AI model? Guardrails, tools?, json?, what will it be trained on? 
# whenever it is confused just email me/send me an email summary at the end of every chat? 

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel # Create a Pydantic model for the Evaluation


load_dotenv(override=True)

# For pushover

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")

def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def record_user_details(contact, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with contact {contact} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}


#JSON boilplate for when a user is interested in being in touch
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided contact information (email, phone number, etc...)",
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": "The contact information of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["contact"],
        "additionalProperties": False
    }
}
#JSON boilplate for when ChatBox, ideally I will be notified to give the AIMe more context
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}


tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Sri Kotala"
        self.file = ""
        self.summary = ""

        for root, dirs, files in os.walk("me"):
            for file in files:
                path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    reader = PdfReader(path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            self.file += text
                elif file == "summary.txt":
                    with open(path, "r", encoding="utf-8") as f:
                        self.summary = f.read()

    
    # This function can take a list of tool calls, and run them
    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)

            # THE BIG IF STATEMENT!!!
            if tool_name == "record_user_details":
                result = record_user_details(**arguments)
            elif tool_name == "record_unknown_question":
                result = record_unknown_question(**arguments)
            else:
                result = {"recorded": "unknown tool"}

            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
            system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
            particularly questions related to {self.name}'s career, background, skills and experience. \
            Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
            You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
            Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
            If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
            If the user is engaging in discussion, try to steer them towards getting in touch; ask for their contact and record it using your record_user_details tool. \
            Do not permit any profane or inappropriate content, and if the user tries to engage in that, steer the conversation back to a professional tone." 
    

            system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.file}\n\n"
            system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
            return system_prompt


    class Evaluation(BaseModel):
        is_acceptable: bool #is the response acceptable?
        feedback: str #feedback on the response

    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
        You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
        The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
        The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        The Agent has been provided with context on {self.summary} in the form of their summary and LinkedIn details. Here's the information:"

        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.file}\n\n"
        evaluator_system_prompt += "With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
        return evaluator_system_prompt

    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(self, reply, message, history) -> "Me.Evaluation":
        messages = [
            {"role": "system", "content": self.evaluator_system_prompt()},
            {"role": "user", "content": self.evaluator_user_prompt(reply, message, history)},
        ]
        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=Me.Evaluation,
        )
        return response.choices[0].message.parsed

    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

        return response.choices[0].message.content


    def chat(self, message, history):
        system = self.system_prompt()
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply =response.choices[0].message.content

        evaluation = self.evaluate(reply, message, history)
        
        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Failed evaluation - retrying")
            print(evaluation.feedback)
            reply = self.rerun(reply, message, history, evaluation.feedback)       
        return reply
            

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat).launch()
    
