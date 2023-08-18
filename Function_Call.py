import os
import time
import json
import ast
import pandas as pd
from scipy import spatial
from tqdm.auto import tqdm

import openai
import tiktoken
import pinecone,chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import(
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma,Pinecone
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import (
    RetrievalQA,
    ConversationChain,
    ConversationalRetrievalChain
    )
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage
)
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
    )

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
    )

from langchain.callbacks import get_openai_callback
start_time = time.time()

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-16k"
EMBEDDING_MODEL = "text-embedding-ada-002"
csv_path = "/Volumes/work/Project/AIGC/OpenAI/Function_Call/data/FIFA_World_Cup_2022.csv"
persist_directory = '/Volumes/work/Project/AIGC/Langchain/docs/chroma_22b/'


pinecone_status=True
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") 
pinecone_new_index_name = 'cross-the-great-river'



class FunctionRunner:
    def __init__(self, api_key, frame: str = None, vectorstore: str = None):
        openai.api_key = api_key
        self.api_key = api_key
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.GPT_MODEL = GPT_MODEL
        self.FRAMEWORK = frame
        self.VECTORSTORES = vectorstore
        self.df = None
    
    # prepare three candidate functions for openai's function calling.
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
            "forecast": ["sunny"],
        }
        return json.dumps(forecast_info)
    
    def ask(self, query: str) -> str:
        if self.FRAMEWORK == 'langchain':
            return self.langchain_ask(query)
        elif self.FRAMEWORK == 'llamaindex':
            return self.llamaindex_ask
        else:
            return self.raw_ask(query)
        
    def raw_ask(self,query: str) -> str:
        if self.VECTORSTORES == 'pinecone':
            return self.ask_pinecone(query)
        elif self.VECTORSTORES == 'chroma':
            return self.ask_chroma(query)
        elif self.VECTORSTORES is None:
            return self.ask_openai(query)
        else:
            raise ValueError('Invalid vectorstores value')
        
    def langchain_ask(self,query: str) -> str:
        embedding = OpenAIEmbeddings()
        chat_model = ChatOpenAI(model=self.GPT_MODEL,temperature=0,max_tokens=256)
        memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
        if self.VECTORSTORES == 'pinecone':
            pinecone.init(
                api_key=pinecone_api_key,
                environment=pinecone_env
                )
            index = pinecone.Index(pinecone_new_index_name)
            vectordb = Pinecone(index,embedding.embed_query,'text')
        elif self.VECTORSTORES == 'chroma':
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
                )
        retriever = vectordb.as_retriever(
            search_tpye='similarity',
            search_kwargs={'k':1}
            )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory
            )
        with get_openai_callback() as cb:
            answer = qa_chain(query)

        content = answer['answer']
        prompt_tokens = cb.prompt_tokens
        completion_tokens = cb.completion_tokens
        total_tokens = cb.total_tokens

        response = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        print(response["choices"][0]["message"]["content"])
        return response

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

    def ask_openai(self, query: str, csv_path: str = None, df_cache: pd.DataFrame = None, token_budget: int = 4096 - 500) -> str:
        if csv_path is None:
            csv_path = csv_path
            
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
        return response 
    
    
    def ask_pinecone(self,query: str,limit = 12000):
        pinecone.init(
            api_key = pinecone_api_key,
            environment = pinecone_env
        )

        # connect to index
        index = pinecone.Index(pinecone_new_index_name)

        res = openai.Embedding.create(
            input=[query],
            model=self.EMBEDDING_MODEL
        )

        # retrieve from Pinecone
        xq = res['data'][0]['embedding']

        # get relevant contexts
        res = index.query(xq, top_k=2, include_metadata=True)
        contexts = [
            x['metadata']['text'] for x in res['matches']
        ]
        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the question based on the context below.\n\n"+
            "Context:\n"
        )
        prompt_end = (
            f"\n\nQuestion: {query}"
        )
        # append contexts until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i]) +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
                )
        response=openai.ChatCompletion.create(
            model=self.GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response
    
    """def ask_chroma(self,query: str) -> str:
        chroma_client = chromadb.Client() #Get Chroma Client
        embedding_function = OpenAIEmbeddingFunction(api_key=self.api_key,model_name=EMBEDDING_MODEL)
        collection = chroma_client.create_collection(name='my_collection') # Collection are where you'll store your embeddings,documents,and any additional metadata.
        
        # Chroma will store your text,and handle tokenization,embedding,and indexing automatically.
        collection.add(
            embeddings=[[1.2,2.3,4.5],[6.7,8.2,9.2]] #you can load embeddings generated by yourself.
            documents=['This is a document','This is another document'],
            metadatas=[{'source': 'my_source'},{'source': 'my_source'}],
            ids=['id1','id2']
            )
        # You can query the collection with a list of query texts.and Chroma will return the n most similar results
        results = collection.query(
            query_texts=[query],
            n_results=2
            )
        return results"""

    def run_function_calling(self, query:str):
        messages = [
            {"role": "system", "content": "You are smart and helpful AI assistant.You only use the functions you have been provided with once the answer cannot be found in your training data."},
            {"role": "user", "content": query},
            ]
        functions = [
        {
            "name": "ask",
            "description": "Answer a query related to the texts and embeddings in vectorstore",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Questions user want assistant to answer using embeddings",
                    },
                },
                "required": ["question"],
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
               function_response = function_to_call(query=function_args.get("question"))
               return function_response
            elif function_name == "get_current_weather":
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                function_call_message = {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                    }
                messages.append(function_call_message)
                return messages
            elif function_name == "get_n_weather_forecast":
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                    num_days=function_args.get("num_days")
                )
                function_call_message = {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                    }
                messages.append(function_call_message)
                return messages
        
            # Test section
            messages.append(function_call_message) # extend conversation with function response
            second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            )  # get a new response from GPT where it can see the function response

# Now you can use the class to call the function
runner = FunctionRunner(openai.api_key,frame='langchain',vectorstore='chroma')
result=runner.run_function_calling("CROSS THE GREAT RIVER的原名是什么?")
print(result)

end_time=time.time()
execution_time = end_time - start_time
print(execution_time)