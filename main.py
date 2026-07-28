from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def main():
    print("Hello from rag!")
    print("Groq and dotenv imports loaded successfully.")


if __name__ == "__main__":
    main()
