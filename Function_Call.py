import ast
import openai
import pandas as pd
import tiktoken
from scipy import spatial
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-0613"

class FunctionRunner:
    def __init__(self, api_key, model, embeddings_path):
        openai.api_key = api_key
        self.EMBEDDING_MODEL = "text-embedding-ada-002"
        self.GPT_MODEL = model
        self.df = None
        self.embeddings_path = embeddings_path

    def get_current_weather(self, location, unit):
        weather_info = {
            "location": location,
            "temperature": "22",
            "unit": unit,
        }
        return json.dumps(weather_info)

    def get_n_weather_forecast(self, location, unit, num_days):
        forecast_info = {
            "location": location,
            "temperature": "30",
            "unit": unit,
            "num_days": num_days,
            "forecast": ["rainy"],
        }
        return json.dumps(forecast_info)

    def strings_ranked_by_relatedness(
        self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        query_embedding_response = openai.Embedding.create(
            model=self.EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(self, text: str, model: str) -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def ask(self, query: str, csv_path: str = None, df_cache: pd.DataFrame = None, token_budget: int = 4096 - 500) -> str:
        if csv_path is None:
            csv_path = self.embeddings_path
            
        if df_cache is None:
            self.df = pd.read_csv(csv_path)
            self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        else:
            self.df = df_cache

        message = self.query_message(query, self.df, token_budget=token_budget)
        messages = [{"role": "user", "content": message}]
        response = openai.ChatCompletion.create(
            model=self.GPT_MODEL,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message 

    def query_message(self, query: str, df: pd.DataFrame, token_budget: int) -> str:
        strings, relatednesses = self.strings_ranked_by_relatedness(query, df)
        introduction = 'You are helpful AI assistant, If the answer cannot be found in your training data, Use the below articles to answer the subsequent question. "'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
            if (
                self.num_tokens(message + next_article + question, model=self.GPT_MODEL)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question

    def run_function_calling(self, query:str):
        messages = [{"role": "user", "content": query}]
        functions = [
        {
            "name": "ask",
            "description": "Answer a query using GPT and a dataframe of relevant texts and embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Questions user want assistant to answer using embeddings",
                    },
                },
                "required": ["answer"],
            },
        },
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
        {
            "name": "get_n_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state,e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The tempeture unit to use.Infer this from the users location."
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "unit", "num_days"]
            }
        }   
        ]
        
        response = openai.ChatCompletion.create(
            model=self.GPT_MODEL,
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            available_functions = {
                "ask": self.ask,
                "get_current_weather": self.get_current_weather,
                "get_n_weather_forecast": self.get_n_weather_forecast
            }
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            if function_name =="ask":
               function_response = function_to_call(query=function_args.get("answer"))
            elif  function_name =="get_current_weather":
                  function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit")
                  )
            elif  function_name =="get_n_weather_forecast":
                  function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                    num_days=function_args.get("num_days"),
                  )
            function_call_message={
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
            messages.append(function_call_message) 
            return messages

# Now you can use the class to call the function
runner = FunctionRunner(openai.api_key,GPT_MODEL,"/Volumes/work/Project/AIGC/OpenAI/Function_Call/data/FIFA_World_Cup_2022.csv")
result=runner.run_function_calling("What's the weather like in Boston over the next 3 days?")
print(result)