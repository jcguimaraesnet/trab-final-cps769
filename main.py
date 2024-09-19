#python -m venv venv
#source venv/bin/activate
import os
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from typing import Any, Dict, Literal, Optional
#from langchain_core.pydantic_v1 import BaseModel, Field
#from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from pandasql import sqldf
import pandas as pd
from langchain_core.tools import tool

from langchain_core.messages import AIMessage
import json
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from chainlit.input_widget import Select, Switch, Slider

class QueryParameters(BaseModel):
    """Information about parameters."""
    Client: Optional[str] = Field(default=None, description="The name of the client or source server. Only use if explicitly specified. Examples: ba, rj, df, ce, es.")
    Server: Optional[str] = Field(default=None, description="The name of the server or target server. Only use if explicitly specified.")
    StartDate: Optional[str] = Field(default=None, description="The start date of the period. Only use if explicitly specified.")
    EndDate: Optional[str] = Field(default=None, description="The end date of the period. Only use if explicitly specified.")
    UserQuestion: Optional[str] = Field(default=None, description="The query to be answered. Only use if explicitly specified.")
    TemplateQuestion: Optional[str] = Field(default=None, description="The template question to be answered. Only use if explicitly specified.")
    RttIncrease: Optional[float] = Field(default=None, description="The percentage of increase in RTT. Only use if explicitly specified.")
    Language: Optional[str] = Field(default="en-us", description="The language of UserQuestion attribute. Analyze the language of the UserQuestion attribute to fill the abbreviation that indicate the language and region associated with the language variant used. Examples: pt-br, en-us, es-es.")
    QuestionNumber: Literal[1, 2, 3, 4] = Field(
        default=None,
        description="The number of the question. None means that the question is not understood.",
    )

model = ChatOpenAI(openai_api_key = os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-4o-mini",
        streaming=True)

prompt = """You are an expert extraction relevant information from the text.
You are very good at identifying similar questions between Template Question and User Question even if they are in different translations. \
When the template question is not similar to the user's question, fill in the number question field with None.

Template Question: {template_question}
User Question: {query}
"""

question1_template = "Question 1: Which client has the worst video reception quality over time between <StartDate> and <EndDate>?"
question2_template = "Question 2: Which server provides the most consistent QoE between <StartDate> and <EndDate>?"
question3_template = "Question 3: What is the best server switching strategy to maximize quality of experience for client '<Client>' between <StartDate> and <EndDate>?"
question4_template = "Question 4: If latency increases by <RttIncrease>, how does this affect the QoE of client '<Client>' between <StartDate> and <EndDate>?"

embeddings = OpenAIEmbeddings()
prompt_templates = [question1_template, question2_template, question3_template, question4_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def extract_structured_output_prompt(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(prompt).partial(template_question=most_similar)

chain_extract_prompt = (
    {"query": itemgetter("query")}
    | RunnableLambda(extract_structured_output_prompt)
    | model.with_structured_output(schema=QueryParameters)
)

def get_bitrate(query_params: QueryParameters):
    """Get bitrate by QueryParameters."""
    df = pd.read_csv('data/bitrate_train.csv')
    query = f"""SELECT client,
                        server,
                        strftime('%Y-%m-%d %H:%M', datetime(timestamp, 'unixepoch')) as datetime,
                        avg(bitrate) as avg_bitrate
                FROM df
                WHERE strftime('%Y-%m-%d %H:%M', datetime(timestamp, 'unixepoch'))
                    BETWEEN strftime('%Y-%m-%d %H:%M', '{query_params.StartDate}')
                    AND strftime('%Y-%m-%d %H:%M', '{query_params.EndDate}')
                group by client,
                        server,
                        strftime('%Y-%m-%d %H:%M', datetime(timestamp, 'unixepoch'))
            """

    result = sqldf(query)
    return result

def get_rtt(query_params: QueryParameters):
    """Get rtt by QueryParameters."""
    df = pd.read_csv('data/rtt_train.csv')
    query = query = f"""SELECT client,
                               server,
                               rtt,
                               strftime('%Y-%m-%d %H:%M', datetime(timestamp, 'unixepoch')) as datetime,
                               strftime('%Y-%m-%d %H:%M:%S', datetime(timestamp, 'unixepoch')) as datetime_rtt_original
                        FROM df
                        WHERE strftime('%Y-%m-%d %H:%M', datetime(timestamp, 'unixepoch'))
                                BETWEEN strftime('%Y-%m-%d %H:%M', '{query_params.StartDate}')
                                AND strftime('%Y-%m-%d %H:%M', '{query_params.EndDate}')
                        """

    result = sqldf(query)
    if query_params.Client is not None and query_params.RttIncrease is not None:
        result.loc[result['client'] == query_params.Client, 'rtt'] *= ((query_params.RttIncrease / 100) + 1)
    return result

def normalize_data(df, column_name):
    """Normalize the data of a DataFrame."""
    scaler = MinMaxScaler(feature_range=(0.00001, 1))
    df[column_name + '_normalized'] = scaler.fit_transform(df[[column_name]])
    return df

def merge_normalize_dataframe(df1, df2):
    """Merge two DataFrames by datetime."""
    merge = pd.merge(df1, df2, on=['client', 'server', 'datetime'], how='inner')
    merge = normalize_data(merge, "avg_bitrate")
    merge = normalize_data(merge, "rtt")
    return merge

def calculate_qoe(merged_df):
    """Calculate the QoE."""
    merged_df['qoe'] = merged_df['avg_bitrate_normalized'] / merged_df['rtt_normalized']
    return merged_df

@tool
def get_question1(query_params: QueryParameters):
    """Which client has the worst video reception quality."""
    df_bitrate = get_bitrate(query_params)
    df_rtt = get_rtt(query_params)
    merged_df = merge_normalize_dataframe(df_bitrate, df_rtt)
    df = calculate_qoe(merged_df)

    df_grouped = df.groupby(['client'])['qoe'].mean().sort_values(ascending=False).reset_index()
    df_grouped['position'] = range(1, len(df_grouped) + 1)
    return df_grouped

@tool
def get_question2(query_params: QueryParameters):
    """Which server provides the most consistent QoE."""
    df_bitrate = get_bitrate(query_params)
    df_rtt = get_rtt(query_params)
    merged_df = merge_normalize_dataframe(df_bitrate, df_rtt)
    df = calculate_qoe(merged_df)

    df = df.groupby(['server'])['qoe'].var().sort_values(ascending=True).reset_index()
    df['position'] = range(1, len(df) + 1)
    return df

@tool
def get_question3(query_params: QueryParameters):
    """What is the best server switching strategy to maximize quality of experience for client."""

    df_bitrate = get_bitrate(query_params)
    df_rtt = get_rtt(query_params)
    merged_df = merge_normalize_dataframe(df_bitrate, df_rtt)
    df = calculate_qoe(merged_df)

    # step 1
    df = df[df['client'] == query_params.Client]
    df.loc[:, 'datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:00')
    df_1 = df.groupby(['client', 'server'])['qoe'].mean().sort_values(ascending=False).reset_index()
    print(f'\ndf_1: {df_1}\n')

    # step 2 - best QoE client-SERVER pair
    server = df_1['server'].iloc[0]
    print(f'\nserver: {server}\n')

    # step 3 - get QoE client-server baseline
    df_2 = df.groupby(['datetime', 'client', 'server'])['qoe'].mean().reset_index().sort_values(by='datetime', ascending=True)
    df_2 = df_2[df_2['server'] == server]
    print(f'\ndf_2: {df_2}\n')

    # step 4 - get QoE client-server to compare
    df_3 = df.groupby(['datetime', 'client', 'server'])['qoe'].mean().reset_index().sort_values(by='datetime', ascending=True)
    df_3 = df_3[df_3['server'] != server]
    print(f'\ndf_3: {df_3}\n')

    partial_response = """
    Based on json below, generate below:
    1. a title igual a: #### Table 1: Ranking of the highest QoE by client-server
    2. a brief explanation of up to ten words.
    3. a similar markdown table the json below
    {json_table_ranking}

    Based on json below, generate below:
    1. a title igual a: #### Table 2: QoE per hour of baseline client-server pair ('{client}' x '{server}')
    2. a brief explanation of up to ten words.
    3. a similar markdown table the json below
    {json_table_baseline}

    Based on json below, generate below:
    1. a title igual a: #### Table 3: QoE per hour of client-server pairs for compare
    2. a similar markdown table the json below
    {json_table_compare}
    """

    response = partial_response.format(
        client=query_params.Client,
        server=server,
        json_table_ranking=df_1.to_json(orient='records'),
        json_table_baseline=df_2.to_json(orient='records'),
        json_table_compare=df_3.to_json(orient='records'),
    )
    return response

@tool
def get_question4(query_params: QueryParameters):
    """If latency increases by 99%, how does this affect the QoE of client."""
    json_query_params = json.loads(query_params.json())
    df_with_increase = get_question1.invoke({'query_params': json_query_params})

    query_params.RttIncrease = None
    query_params.Client = None
    json_query_params = json.loads(query_params.json())
    df_without_increase = get_question1.invoke({'query_params': json_query_params})

    partial_response = """
    Client: {client}
    RTT Increase: {rtt_increase}%\n\n
    Json of clients without RTT increase:\n{json_without_increase}\n\n
    Json of clients with RTT increase:\n{json_with_increase}\n\n
    """

    response = partial_response.format(
        client=query_params.Client,
        rtt_increase=query_params.RttIncrease,
        json_without_increase=df_without_increase,
        json_with_increase=df_with_increase,
    )
    return response

@tool
def get_noanswer():
    """No answer found for the question."""
    return "No answer found for the question."


#####################################################################
# subchain de function call
tools = [get_question1, get_question2, get_question3, get_question4, get_noanswer]
llm_with_tools_aux = model.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        print(f'\nRouter Function Calling: {tool_call["name"]}\n\n')
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls
chain_router_tools = llm_with_tools_aux | call_tools

#####################################################################


#####################################################################
# question 1 final
final_response_question1 = """You are a expert to resolve the following topic questions:

json of clients sorted by best QoE score. Attention! It is important to consider that the QoE value in the json table is in scientific notation.
{content}

Based on json table provided above, answer the topic below using the response template below replacing the placeholders.
### Main answer:
Client '<client last position>' has the lowest quality of experience (QoE) in video streaming with a QoE score of <qoe last position>.

Based on the description "1. Normalization of bitrate and latency; 2. bitrate divided latency; 3. Average (step 2)", explain in three subtopics how QoE is calculated.
### Step-by-step guide to calculating the QoE score:

Based on the formula "QoE = Average bitrate (kbps) / Latency (sec)", which calculates the quality of experience (QoE) of video transmission between clients and servers, please answer the topic below, without using the provided json, explaining in two subtopics what the minimum and maximum QoE values ​​mean.
### Interpretation of minimum and maximum values

Based on json table provided above, answer the topic below.
### The table below displays clients in order of QoE (highest QoE score):
<markdown_table>
"""
prompt_final_question1 = PromptTemplate.from_template(final_response_question1)
#####################################################################

#####################################################################
# question 2 final
final_response_question2 = """You are a expert to resolve the following topic questions:

json of servers sorted by smallest variances QoE score (most consistent). Attention! It is important to consider that the QoE value in the json table is in scientific or decimal notation.
{content}

Based on json table provided above, answer the topic below using the response template below replacing the placeholders. Translate the response template below to the pt-br language.
### Main answer:
The server '<position 1 of server>' has the smallest variance of quality of experience (QoE) in video streaming with a QoE score of <position 1 of qoe>, in other words, the most consistent QoE among all servers.

Based on json table provided above, answer the topic below.
### Below is the table of servers sorted by position (smallest variances of QoE score):
<markdown_table>
"""
prompt_final_question2 = PromptTemplate.from_template(final_response_question2)
#####################################################################

#####################################################################
# question 3 final
final_response_question3 = """You are a expert to compare tables.

{content}

Generate a new fourth markdown table titled "server change plan" using the second and third tables, comparing the datetime column of each record in the second table with the third table, checking if there is a client-server pair with a higher QoE.

Create this new table with the columns below:
- Date: Column datetime of the second table
- Client: Column client of the second table
- Server: Column server of the second table
- QoE: Column qoe of the second table
- Server*: Column server of the third table
- QoE*: Column qoe of the third table
- Change?: Column with the value "Yes" if the baseline QoE is lower than the highest QoE, otherwise "No".

Please answer by providing all four tables in the same order and a summary explanation with rows where the baseline QoE is lower.
"""
prompt_final_question3 = PromptTemplate.from_template(final_response_question3)
#####################################################################


#####################################################################
# question 4 final
final_response_question4 = """You are a expert to resolve the following topic questions:

{content}

Write a text summarizing the impact of a rtt increase simulation for client informed above using data from the two json tables. In your answer, also consider the two questions below.
1. Compare QoE of the client informed above in both tables (decline percentage)
2. Check if the position of the client informed above with the increase in rtt has changed in relation to the table without an increase in rtt.

Based on json without the increase in RTT provided above, generate a similar markdown table below.
### Table 1: QoE and Client (before RTT Increase)
<markdown_qoe_no_increase_rtt>

Based on json with the increase in RTT provided above, generate a similar markdown table below.
### Table 2: QoE and Client (after RTT Increase)
<markdown_qoe_with_increase_rtt>
"""
prompt_final_question4 = PromptTemplate.from_template(final_response_question4)
#####################################################################

#####################################################################
# no answer final
final_response_noanswer = """
The user has provided a question that you are unable to answer.
Please respond to the user with the exact following answer:

I didn't understand your question.
Please specify the terms of your question better,
considering that I am an expert in answering questions
about bitrate and rtt data between client and server machines.
"""
prompt_noanswer = PromptTemplate.from_template(final_response_noanswer)
#####################################################################


#####################################################################
# subchain para extrair dados
prompt_route_chain = PromptTemplate.from_template("""On based in Template Question parameters below, execute a tool
Parameter: {query_parameters}
""")
chain_data_question_1234 = { "query_parameters": itemgetter("query_parameters") } | prompt_route_chain | chain_router_tools
#####################################################################


#####################################################################
# subchains de resposta final
chain_final_question_1 = {"content": chain_data_question_1234 } | prompt_final_question1 | model | StrOutputParser()
chain_final_question_2 = {"content": chain_data_question_1234 } | prompt_final_question2 | model | StrOutputParser()
chain_final_question_3 = {"content": chain_data_question_1234 } | prompt_final_question3 | model | StrOutputParser()
chain_final_question_4 = {"content": chain_data_question_1234 } | prompt_final_question4 | model | StrOutputParser()
chain_final_noanswer = prompt_noanswer | model | StrOutputParser()
#####################################################################


#####################################################################
# roteamento por custom function (para subchain de resposta final)
def route_final_question(info):
    query_parameters = info['query_parameters']
    print(f'\nQueryParameters: \n{json.loads(query_parameters.json())}\n\n')

    print(f'Router Final Question: {query_parameters.QuestionNumber}\n')
    if query_parameters.QuestionNumber == 1:
        return chain_final_question_1
    elif query_parameters.QuestionNumber == 2:
        return chain_final_question_2
    elif query_parameters.QuestionNumber == 3:
        return chain_final_question_3
    elif query_parameters.QuestionNumber == 4:
        return chain_final_question_4
    else:
        return chain_final_noanswer

route_chain_final_question = {"query_parameters": chain_extract_prompt } | RunnableLambda(route_final_question)
#####################################################################


#####################################################################
# subchain para traduzir a resposta
translate_response = """You are a translation expert.\n\n

Translate the response below to '{language}'.\n
Response: {response}\n

Attention! Answer only the response translated.\n\n\n
"""
chain_translate = {"response": route_chain_final_question, "language": itemgetter("language") } | PromptTemplate.from_template(translate_response) | model | StrOutputParser()
#####################################################################

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Language",
                label="Language",
                values=["pt-br", "en-us", "es-es"],
                initial_index=0,
            ),
        ]
    ).send()
    cl.user_session.set("language", settings["Language"])

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("language", settings["Language"])

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="1. Cliente com pior QoE?",
            message="Qual cliente tem pior qualidade de experiência de vídeo entre 2024-06-07 00:00 e 2024-06-07 03:00?",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="2. Servidor mais consistente",
            message="Qual servidor fornece a QoE mais consistente entre 2024-06-07 00:00 e 2024-06-07 03:00?",
            icon="/public/idea.svg",
            ),
        cl.Starter(
            label="3. Plano de mudança de servidor",
            message="Qual é a melhor estratégia de troca de servidor para maximizar a qualidade de experiência do cliente 'ba' entre 2024-06-07 00:00 e 2024-06-07 03:00?",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="4. Aumento na latência",
            message="Se a latência aumentar 20%, como isso afeta a QoE do cliente 'ba' entre 2024-06-07 00:00 e 2024-06-07 03:00?",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="Qual cliente windows 10?",
            message="Qual cliente tem sistema operacional windows?",
            icon="/public/terminal.svg",
            ),
        ]

@cl.on_message
async def on_message(message: cl.Message):

    language = cl.user_session.get("language")

    # chain de entrypoint
    output_translate = chain_translate.invoke({"query": message.content, "language": language})

    await cl.Message(content=output_translate).send()
