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

    st.title("CSV Querying with LangChain & Streamlit")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file to the data folder
        data_folder = "./data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Read the uploaded CSV into a DataFrame
            panda_df = pd.read_csv(file_path).fillna(0)

            st.write("### Dataset Preview")
            st.write(panda_df.head())

            # # Display basic statistics
            # st.write("### Data Summary")
            # st.write(panda_df.describe())

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
        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")

if __name__ == "__main__":
    main()