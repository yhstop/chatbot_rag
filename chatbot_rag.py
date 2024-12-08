import streamlit as st

import openai
import os
import pandas as pd
import ast

from dotenv import load_dotenv

import numpy as np
from numpy import dot
from numpy.linalg import norm

from streamlit_chat import message

# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model='text-embedding-ada-002'
    )
    return response.data[0].embedding

# folder_path와 file_name을 결합하여 file_path = './data/embedding.csv'
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# embedding.csv 파일을 읽어서 DataFrame으로 변환
df = pd.read_csv(file_path)
# 'embedding' 열의 각 값을 문자열에서 리터럴로 변환
df['embedding'] = df['embedding'].apply(ast.literal_eval)


# 주어진 질의로부터 유사한 문서 개를 반환하는 검색 시스템.
# 함수 return_answer_candidate내부에서 유사도 계산을 위해 cos_sim을 호출.
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

# 챗봇의 답변을 만들기 위해 사용될 프롬프트를 만드는 함수.
def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    system_role = f"""You are an artificial intelligence language model that specializes in summarizing \
    and answering documents about Seoul's youth policy.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document: 
            doc 1 :{str(result.iloc[0]['text'])}
            doc 2 :{str(result.iloc[1]['text'])}
            doc 3 :{str(result.iloc[2]['text'])}
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ] 
    return messages

# 위의 create_prompt 함수가 생성한 프롬프트로부터 챗봇의 답변을 만드는 함수.
def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=1024)
    return result.choices[0].message.content


# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 질문을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

# submitted의 값이 True면 챗봇이 답변을 하기 시작
if submitted and user_input:
    # 프롬프트 생성
    prompt = create_prompt(df, user_input)
    
    # 생성한 프롬프트를 기반으로 챗봇 답변을 생성
    chatbot_response = generate_response(prompt)

    # 화면에 보여주기 위해 사용자의 질문과 챗봇의 답변을 각각 저장
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_response)

# 챗봇의 답변이 있으면 사용자의 질문과 챗봇의 답변을 가장 최근의 순서로 화면에 출력
if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))