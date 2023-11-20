import streamlit as st
from langchain.llms import ctransformers

llm = ctransformers(
    model = 'llama-2-7b-chat.ggmlv3.q2_K.bin',
    model_type = 'llama'
)

st.title('인공지능 시인')

content =  st.text_input('시의 주제를 제시해주세요.')

st.write('시의 주제는 ' + content + ' 입니다.')

if st.button('시 작성 요청하기'):
        with st.spinner('시를 작성중입니다...'):
            result = llm.predict(content + "에 대한 시를 써줘")
            st.write(result)





