# QueryPilot

QueryPilot is a Streamlit application that allows users to interact with a pandas DataFrame using a language model (LangChain LLM) to answer questions about the data. The application provides a user interface where users can either select a predefined question from a dropdown menu or write their own custom question. The application then processes the question using the language model and displays the result. If the result contains tabular data, it also provides an option to download the data as a CSV file.

## Key Components

### Imports and Environment Setup
- The program imports necessary libraries such as `os`, `pandas`, `streamlit`, `dotenv`, and `langchain_openai`.
- It loads environment variables from a `.env` file using `load_dotenv()`.

### Main Function
- The `main()` function is defined to encapsulate the main logic of the application.

### Load Environment Variables
- The program loads the OpenAI API key from the environment variables. If the API key is missing, it displays an error message and stops execution.

### Initialize LangChain LLM
- The program initializes the LangChain language model (`gpt-3.5-turbo`) using the `ChatOpenAI` class.

### Upload CSV into a DataFrame
- The program allows users to upload a CSV file into a pandas DataFrame and fills any missing values with 0. If the CSV file is not found, it displays an error message and stops execution.

### Streamlit UI
- The program sets up the Streamlit user interface:
  - Displays the title and a preview of the dataset.
  - Provides a dropdown menu for selecting a predefined question.
  - Provides a text input field for writing a custom question.
  - A button to run the query.

### Query Processing
- When the "Run Query" button is clicked, the program constructs a query using the selected or custom question and predefined prompt prefixes and suffixes.
- It invokes the language model agent with the constructed query.

### Display Results
- The program checks the type of the result:
  - If the result contains both a DataFrame and text, it displays the text and the DataFrame, and provides a download button for the DataFrame.
  - If the result is a DataFrame, it displays the DataFrame and provides a download button.
  - If the result is text, it displays the text using `st.markdown`.

### Run the Application
- The `main()` function is called if the script is run as the main module.

## Example Usage
- Users can upload a CSV file and select a question like "What is the average salary by department?" from the dropdown menu or write their own question.
- The application processes the question and displays the result.
- If the result includes tabular data, users can download it as a CSV file for future reference.

## How to Run the Application

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/QueryPilot.git
    cd QueryPilot
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Rename the `.env.example` file to `.env`.
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```

5. **Run the Streamlit application**:
    ```sh
    streamlit run csv_agent.py
    ```

6. **Open your web browser**:
    - Navigate to `http://localhost:8501` to interact with the application.

This program provides an interactive way to explore and analyze data using natural language queries, leveraging the capabilities of a language model and pandas DataFrame operations.