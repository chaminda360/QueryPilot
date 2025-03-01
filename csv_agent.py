import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

load_dotenv()

def main():
    # Load environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("⚠️ OPENAI_API_KEY is missing. Check your .env file!")
        return

    # Initialize LangChain LLM
    llm_name = "gpt-3.5-turbo"
    model = ChatOpenAI(model=llm_name)

    # Read CSV into a DataFrame
    csv_path = "./data/salaries_2023.csv"

    if not os.path.exists(csv_path):
        st.error(f"⚠️ CSV file not found at {csv_path}!")
        return
    else:
        panda_df = pd.read_csv(csv_path).fillna(0)

    st.title("CSV Querying with LangChain & Streamlit")

    st.write("### Dataset Preview")
    st.write(panda_df.head())

    # Create agent for csv file and pandas dataframe with verbose mode on
    agent = create_pandas_dataframe_agent(
        llm=model, # model
        df=panda_df, # pandas dataframe
        verbose=True, # verbose
    )

    # Define the prompt prefix and suffix
    CSV_PROMPT_PREFIX = """
    First set the pandas display options to show all the columns,
    get the column names, then answer the question.
    """

    CSV_PROMPT_SUFFIX = """
    - **ALWAYS** before giving the Final Answer, try another method.
    Then reflect on the answers of the two methods you did and ask yourself
    if it answers correctly the original question.
    If you are not sure, try another method.
    FORMAT 4 FIGURES OR MORE WITH COMMAS.
    - If the methods tried do not give the same result,reflect and
    try again until you have two methods that have the same result.
    - If you still cannot arrive to a consistent result, say that
    you are not sure of the answer.
    - If you are sure of the correct answer, create a beautiful
    and thorough response using Markdown.
    - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
    ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
    - **ALWAYS**, as part of your "Final Answer", explain how you got
    to the answer on a section that starts with: "\n\nExplanation:\n".
    In the explanation, mention the column names that you used to get
    to the final answer.
    """

    # List of questions to ask the agent
    questions = [
        "how many rows are there in the dataframe?",
        "how many columns are there in the dataframe?",
        "what are the column names in the dataframe?",
        "what is the mean salary in the dataframe?",
        "what is the maximum salary in the dataframe?",
        "what is the minimum salary in the dataframe?",
        "how many unique job titles are there in the dataframe?",
        "what is the total sum of salaries in the dataframe?",
        "what is the average salary for each job title?",
        "how many rows have a salary greater than 100,000?",
        "what are the details of employees with the job title 'Data Scientist'?",
        "what is the average salary by department?",
        "what is the total salary by department?"
    ]

    # Streamlit UI
    st.write("### Ask a Question")
    # Dropdown menu for selecting a question
    selected_question = st.selectbox("Select a question to ask the agent:", questions)
    # Text input for custom question
    custom_question = st.text_input("Or write your own question:")

    if st.button("Run Query"):
        # Use custom question if provided, otherwise use selected question
        question_to_ask = custom_question if custom_question else selected_question

        QUERY = CSV_PROMPT_PREFIX + question_to_ask + CSV_PROMPT_SUFFIX
        res = agent.invoke(QUERY)
        
        st.write("### Final Answer")
        
        # Check if the result contains both a DataFrame and text
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
        elif isinstance(res, pd.DataFrame):
            st.dataframe(res)
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='query_result.csv',
                mime='text/csv',
            )
        else:
            st.markdown(res["output"])

if __name__ == "__main__":
    main()