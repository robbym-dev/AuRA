import openai
from openai import OpenAI
from together import Together
import anthropic
import requests
import time
import os 

class LLMQuery:
    def __init__(self, openai_api_key, together_ai_api_key, anthropic_api_key):
        self.openai_api_key = openai_api_key
        self.together_ai_api_key = together_ai_api_key
        self.anthropic_api_key = anthropic_api_key

    def query_gpt4(self, prompt, system_prompt):
        client = OpenAI()
        
        for attempt in range(5): 
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying in 60 seconds...")
                time.sleep(60)
        return None

    def query_llama(self, prompt, system_prompt):
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        client = Together(api_key=TOGETHER_API_KEY)
         
        for attempt in range(5): 
            try:
                chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        model="meta-llama/Llama-3-70b-chat-hf",
                        stream=True
                    )
                responses = [chunk.choices[0].delta.content for chunk in chat_completion]
                final_response = " ".join(responses)
                return final_response.strip()
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying in 60 seconds...")
                time.sleep(60)
        return None

    def query_claude_opus(self, prompt, system_prompt):        
        responses = []
        
        client = anthropic.Anthropic()

        for attempt in range(5):
            try:
                with client.messages.stream(
                    max_tokens=1024, 
                    system=system_prompt.strip(),
                    messages=[{"role": "user", "content": prompt.strip()}], 
                    model="claude-3-opus-20240229"
                ) as stream:
                    for text in stream.text_stream:
                        responses.append(text)

                final_response = " ".join(responses)
                return final_response.strip()
            except anthropic.APIStatusError as e:
                if 'invalid_request_error' in str(e):
                    print(f"Content filtering policy error: {e}")
                    break
                else:
                    print(f"APIStatusError: {e}")
                    time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"RequestException: {e}")
                time.sleep(1)
            except httpx.RemoteProtocolError as e:
                print(f"RemoteProtocolError: {e}")
                print("Retrying in 60 seconds...")
                time.sleep(60)
        return "No answer generated due to an error or content filtering."
