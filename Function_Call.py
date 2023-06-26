import os
import openai
import json
import ast   #for converting embeddings saved as strings back to arrays
import pandas as pd   #for storing text and embeddings data
import tiktoken   #for counting tokens
from scipy import spatial   #for calculating vector similarities for search

openai.api_key=os.getenv("OPENAI_API_KEY")

#models
EMBEDDING_MODEL="text-embedding-ada-002"
GPT_MODEL="gpt-3.5-turbo-0613"

# download pre-chunked text and pre-computed embeddings
# this file is ~200 MB, so may take a minute depending on your connection speed
embeddings_path = "/Volumes/work/Project/AIGC/OpenAI/Function_Call/data/FIFA_World_Cup_2022.csv"

df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)
# the dataframe has two columns: "text" and "embedding"

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

"""
# examples
strings, relatednesses = strings_ranked_by_relatedness("2022 Qatar World Cup Champion", df, top_n=3)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)
"""

"""
With the search function above, we can now automatically retrieve relevant knowledge and insert it into messages to GPT.
Below, we define a function ask that:
1.Takes a user query
2.Searches for text relevant to the query
3.Stuffs that text into a message for GPT
3.Sends the message to GPT
5.Returns GPT's answer
"""

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text)) 

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

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
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
# set print_message=True to see the source text GPT was working off of
# print(ask('2022年卡塔尔世界杯决赛点球大战的比分是多少？谁获得了最后的冠军？'))

#Example dummy function hard coded to return the same weather
#In production ,this could be your backend API or an external API
def get_current_weather(location,unit):
    """Get the current weather in a given location"""
    weather_info={
        "location":location,
        "temperature":"22",
        "unit":unit,
    }
    return json.dumps(weather_info)

def get_n_weather_forecast(location,unit,num_days):
    """Get an N-day weather forecast"""
    forecast_info={
        "location":location,
        "temperature":"30",
        "unit":unit,
        "num_days": num_days,
        "forecast":["rainy"],
    }
    return json.dumps(forecast_info)

def run_conversation():
    #Step1:send the conversation and available functions to GPT
    messages = [
        {"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
        {"role": "user", "content": "What's the weather like in beijing?"}
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
                "required": ["location","unit"],
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
                        "enum": ["celsius","fahrenheit"],
                        "description": "The tempeture unit to use.Infer this from the users location."
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location","unit","num_days"]
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

    #Step 2:check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        
        available_functions = {
            "ask":ask,
            "get_current_weather": get_current_weather,
            "get_n_weather_forecast": get_n_weather_forecast,
        }  # only one function in this example, but you can have multiple
        
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]

        function_args = json.loads(response_message["function_call"]["arguments"])
        if function_name == "ask":
             function_response = ask()
        elif function_name == "get_current_weather":
             function_response = fuction_to_call(
                 location=function_args.get("location"),
                 unit=function_args.get("unit")
             )
        elif function_name == "get_n_weather_forecast":
             function_response = fuction_to_call(
                 location=function_args.get("location"),
                 unit=function_args.get("unit"),
                 num_days=function_args.get("num_days")
             )
        print(function_name)  #debug printer
        # Step 4: send the info on the function call and function response to GPT
        #messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        second_response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        #return second_response['choices'][0]['message']['content']
        user_facing_message=second_response["choices"][0]["message"]["content"]
        return user_facing_message
        
print(run_conversation())