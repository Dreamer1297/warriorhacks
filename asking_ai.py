from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv() #example of how it might go
nodes = [
   "child", "child", "cake", "person", "dog"
]

bounding_boxes = [
    
]


client = OpenAI(api_key=os.getenv("API_KEY"))

messages = [
    {"role":"system", "content": f'''You describe, in 1-3 sentences, and as concisely as possible, what is happening in an image given
     data that consists of two node groups, which contain at least one objects or entities and are represented by tuples. 
     The one at index 0 will be the subject of the action, and the one at index 1 will be the object of the action. You will first be given
     these nodes. After that, your task is to summarize the scene based on the nodes you have been given in your messages. For each of these groups,
     you must also infer the MOST LIKELY connection, which is an action such as eating throwing walking, for example for an input " (["Person"], ["Dog"]) " you
     output "A person is walking a dog". '''},
     {"role":"system", "content": f"Here are the node-edge connections: {nodes}"},
     {"role":"user", "content": f"Summarize what is happening."}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

print(response.choices[0].message.content)
