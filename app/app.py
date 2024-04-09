import streamlit as st
import pandas as pd
import yfinance as yf
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Function to download cryptocurrency data
def download_crypto_data(ticker, start_date, end_date):
    """
    Downloads cryptocurrency data using yfinance.

    Parameters:
        ticker (str): Ticker symbol of the cryptocurrency (e.g., 'BTC-USD').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: DataFrame containing cryptocurrency data.
    """
    try:
        # Download data using yfinance
        crypto_data = yf.download(ticker, start=start_date, end=end_date)
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Save data to database
        crypto_data.to_csv(f'data/{ticker}.csv')
        return crypto_data
    except Exception as e:
        print("Error occurred:", e)
        return None


def main():
    # Streamlit UI
    st.title("Cryptocurrency Data Analysis")

    # Sidebar inputs
    st.sidebar.header("Download Data")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="BTC-USD")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-01-01'))

    # Download data button
    if st.sidebar.button("Download Data"):
        crypto_data = download_crypto_data(ticker, start_date, end_date)
        if crypto_data is not None:
            st.success(f"Data downloaded successfully for {ticker}.")

    # Load data
    if os.path.exists(f"data/{ticker}.csv"):
        df = pd.read_csv(f"data/{ticker}.csv")
        st.header("Cryptocurrency Data")

        # Checkbox to hide/show data
        hide_data = st.checkbox("Hide Data")
        if not hide_data:
            st.write(df)

        # Load API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")

        # Check if API key is set
        if not api_key:
            st.error("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        else:
            # create Agent AI
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
            agent_executor = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True,
            )

            # Query input
            query_input = st.text_input("Ask a question about the data:", value="How many columns are there?")

            # Run query button
            if st.button("Run Query"):
                query_output = agent_executor.invoke(input=query_input)
                st.write("Query:", query_input)
                st.write("Answer:", query_output['output'])
    else:
        st.warning("Data not available. Please download data first.")

if __name__ == "__main__":
    main()
