import os
import openai
import json
openai.api_key=os.getenv("OPENAI_API_KEY")


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
        {"role": "user", "content": "What's the weather like in Dalian over the next 5 days?"}
        ]

    functions = [
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
        model="gpt-3.5-turbo-0613",
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
            "get_current_weather": get_current_weather,
            "get_n_weather_forecast": get_n_weather_forecast
        }  # only one function in this example, but you can have multiple
        
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]

        function_args = json.loads(response_message["function_call"]["arguments"])

        if function_name == "get_current_weather":
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
        print(function_response)  #debug printer
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
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        #return second_response['choices'][0]['message']['content']
        user_facing_message=second_response["choices"][0]["message"]["content"]
        return user_facing_message
        
print(run_conversation())