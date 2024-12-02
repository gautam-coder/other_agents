import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException,APIRouter
from pydantic import BaseModel
from pptx import Presentation
import os
import requests
from openai import AzureOpenAI
import json
from io import BytesIO
import asyncio
import json
import time
from tavily import TavilyClient
import re
import pdfplumber
from pptx import Presentation
from docx import Document
from io import BytesIO
import base64
# Initialize FastAPI app

app = FastAPI()

# Define request schema
class StepRequest(BaseModel):
    userId: str
    threadId: str
    industry: str
    agent: str
    useCase: str
    step: int
    data: dict

@app.post("/process-step")
async def process_step(request: StepRequest):
    # Fetch the appropriate prompt
    conversation_history = load_conversation_from_file(request.userId,request.threadId)
    prompt = get_prompt(request.industry, request.agent, request.useCase, request.step)

    if prompt == "Prompt not found. Please check the input values.":
        return {"error": prompt}

    # Generate AI response
    ai_response = call_azure_openai(prompt, request.data["user_input"], conversation_history)
    conversation_history.append({"user": request.data["user_input"], "assistant": ai_response})

    # Save conversation history back to the file
    save_conversation_to_file(request.userId,request.threadId, conversation_history[-1])

    # Return the response
    return {
        "userId": request.userId,
        "threadId": request.threadId,
        "step": request.step,
        "prompt": prompt,
        "response": ai_response
    }

# Load the JSON structure
with open("industry_prompts.json", "r") as f:
    industry_prompts = json.load(f)

def get_prompt(industry, agent, use_case, step):
    try:
        # Load the base prompt
        prompt = industry_prompts[industry][agent][use_case]
        prompt = prompt.replace("[industry]", industry).replace("[agent]", agent).replace("[useCase]", use_case)
        full_prompt = f"{prompt}\n\n This conversation is currently in stage {step}, so respond accordingly."
        return full_prompt
    except KeyError:
        return "Prompt not found. Please check the input values."

def call_azure_openai(prompt, context, conversation_history):
    os.environ["AZURE_OPENAI_API_KEY"] = ""
    
    client = AzureOpenAI(
        azure_endpoint = "https://webcrawl.openai.azure.com/",
        api_key=os.getenv(""),
        api_version="2024-02-15-preview"
    )

    message_text = [{"role": "system", "content": prompt}]
    for entry in conversation_history:
        message_text.append({"role": "user", "content": entry["user"]})
        message_text.append({"role": "assistant", "content": entry["assistant"]})
    
    message_text.append({"role": "user", "content": context})

    completion = client.chat.completions.create(
        model="gpt4o", # model = "deployment_name"
        messages = message_text,
        temperature=0.6,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0,
        stop=None
    )
    return completion.choices[0].message.content


# Path to the JSON file where conversations are saved
CONVERSATION_FILE = "conversations.json"




def chatbot_filtration(siteslink):
    os.environ["AZURE_OPENAI_API_KEY"] = ""
    
    client = AzureOpenAI(
        azure_endpoint = "https://webcrawl.openai.azure.com/",
        api_key=os.getenv(""),
        api_version="2024-02-15-preview"
    )

    prompt=f'''Classify search results into categories only the main (e.g., Website, Publication, Scholar, Government) 
    Here is the {siteslink}
    Give it into the kye value pair like'''+'{"website":[],"goverment":[],"organisation":[]...}'+'''
    Only Output format not write any other thing direclty give json Don't mention json :- 
    '''

    message_text = [{"role":"system","content":prompt},{"role":"user","content":"Give me the classification of this sites"}]

    completion = client.chat.completions.create(
        model="web_gpt4", # model = "deployment_name"
        messages = message_text,
        temperature=0.9,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return completion.choices[0].message.content




def tavily_search(querymain):
    tavily_client = TavilyClient(api_key="")
    # Step 2. Executing a context search query
        #context = tavily_client.get_search_context(query="recent merger and acquistion in Ai biochips")
    response=[]
    for query in querymain[:2]:
        response.append(tavily_client.search(query,max_results=20,search_depth="advanced"))
    tt=[]
    links=[]
    print(links)
    for item in response:
    # Loop over the 'results' inside each dictionary
        for result in item['results']:
            links.append({"Citation": result['url'], "content": str(result['content'])})

    if links:
        for i in links:
            tt.append(i['Citation'])
        ab=chatbot_filtration(tt)
        return ab.choices[0].message.content

    else:
        return "we are finding the sources"


#print(tavily_search("who is the founder of google"))

def save_conversation_to_file(userId, threadId, conversation):
    try:
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, "r+") as file:
                data = json.load(file)

                # If userId exists, check for threadId
                if userId in data:
                    if threadId in data[userId]:
                        # Append conversation to the existing thread
                        data[userId][threadId].append(conversation)
                    else:
                        # Create a new thread with the conversation
                        data[userId][threadId] = [conversation]
                else:
                    # Create a new user entry with the thread and conversation
                    data[userId] = {threadId: [conversation]}

                # Rewind the file and save the updated data
                file.seek(0)
                json.dump(data, file, indent=4)
        else:
            # Create a new file with the initial data structure
            with open(CONVERSATION_FILE, "w") as file:
                json.dump({userId: {threadId: [conversation]}}, file, indent=4)

    except Exception as e:
        print(f"Error saving conversation: {e}")

# Function to load the conversation history for a specific userId and threadId
def load_conversation_from_file(userId, threadId):
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as file:
            try:
                data = json.load(file)

                # Validate that the structure is correct
                if not isinstance(data, dict):
                    raise ValueError("Corrupted data: Expected a dictionary at the root level")

                # Safely retrieve the conversation history
                return data.get(userId, {}).get(threadId, [])
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading conversation: {e}")
                return []
    return []