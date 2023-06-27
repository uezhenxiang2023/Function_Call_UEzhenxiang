import os
import openai
import json
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

openai.api_key = os.getenv("OPENAI_API_KEY")

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-0613"

# download pre-chunked text and pre-computed embeddings
# this file is ~200 MB, so may take a minute depending on your connection speed
embeddings_path = "/Volumes/work/Project/AIGC/OpenAI/Function_Call/data/FIFA_World_Cup_2022.csv"

df = None  # DataFrame initialization

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
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

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text)) 

def ask(
    query: str,
    csv_path: str = embeddings_path,
    df_cache: pd.DataFrame = None,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    global df  # reference the global DataFrame variable

    if df_cache is None:
        # Load the DataFrame from the CSV file
        df = pd.read_csv(csv_path)
        # Convert embeddings from CSV str type back to list type
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        # Use the cached DataFrame
        df = df_cache

    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [{"role": "user", "content": message}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'You are helpful AI assistant, If the answer cannot be found in your training data, Use the below articles to answer the subsequent question. "'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "22",
        "unit": unit,
    }
    return json.dumps(weather_info)


def get_n_weather_forecast(location, unit, num_days):
    """Get an N-day weather forecast"""
    forecast_info = {
        "location": location,
        "temperature": "30",
        "unit": unit,
        "num_days": num_days,
        "forecast": ["rainy"],
    }
    return json.dumps(forecast_info)


def run_conversation():
    # Step1: send the conversation and available functions to GPT
    messages = [
        {"role": "system", "content": "回答问题时，请跟客户提问所用的语言保持一致。比如用户用中文提问，你也用中文回答。"},
        {"role": "user", "content": "2022年卡塔尔世界杯决里点球大战的最终比分是多少？最后冠军是谁"}
        ]

    functions = [
        {
            "name": "ask",
            "description": "Answers a query using GPT and a dataframe of relevant texts and embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Questions user want assistant to answer using embeddings",
                    },
                },
                "required": ["query"],
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
        model=GPT_MODEL,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function and pass the relevant arguments
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        if function_name == "ask":
            query = function_args["query"]
            user_facing_message = f"{ask(query)}"  # Call the modified 'ask' function
        else:
            if function_name == "get_current_weather":
                location = function_args["location"]
                unit = function_args["unit"]
                function_response = get_current_weather(location, unit)
            elif function_name == "get_n_weather_forecast":
                location = function_args["location"]
                unit = function_args["unit"]
                num_days = function_args["num_days"]
                function_response = get_n_weather_forecast(location, unit, num_days)
            
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response

            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )  # get a new response from GPT where it can see the function response
            user_facing_message = second_response["choices"][0]["message"]["content"]
    else:
        user_facing_message = response_message["content"]

    return user_facing_message

print(run_conversation())
