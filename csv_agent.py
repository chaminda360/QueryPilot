import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from file_utils import upload_and_save_file, load_csv
from llm_utils import load_api_key, initialize_llm
from prompt_utils import define_prompts,default_questions

def create_agent(model, panda_df):
    return create_pandas_dataframe_agent(llm=model, df=panda_df, verbose=True)

def ask_question_to_agent(agent, panda_df, CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX):
    questions = default_questions()

    st.write("### Ask a Question")
    selected_question = st.selectbox("Select a question to ask the agent:", questions)
    custom_question = st.text_input("Or write your own question:")
    
    if st.button("Run Query"):
        question_to_ask = custom_question if custom_question else selected_question
        QUERY = CSV_PROMPT_PREFIX + question_to_ask + CSV_PROMPT_SUFFIX
        try:
            res = agent.invoke(QUERY)
            st.write("### Final Answer")
            
            if isinstance(res, dict) and "dataframe" in res and "text" in res:
                st.write(res["text"])
                st.dataframe(res["dataframe"])
                csv = res["dataframe"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='query_result.csv',
                    mime='text/csv',
                )
            elif isinstance(res,  panda_df.DataFrame):
                st.dataframe(res)
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='query_result.csv',
                    mime='text/csv',
                )
            else:
                st.markdown(res)
        except Exception as e:
            st.error(f"Error querying the agent: {e}")

def main():
    try:
        api_key = load_api_key()
        if not api_key:
            return

        model = initialize_llm(api_key)
        st.title("CSV Querying with LangChain & Streamlit")
        
        file_path = upload_and_save_file()
        if not file_path:
            return
        
        panda_df = load_csv(file_path)
        if panda_df is None:
            return
        
        CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX = define_prompts()
        agent = create_agent(model, panda_df)
        ask_question_to_agent(agent, panda_df, CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()