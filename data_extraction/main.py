from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from loguru import logger
from typing import List


class Date(BaseModel):
    """
    Represents a date with day, month, and year.
    """

    day: int = Field(..., description="Day of the month")
    month: int = Field(..., description="Month of the year")
    year: int = Field(..., description="Year")


class Dates(BaseModel):
    """
    Represents a list of dates (Dates class).
    """

    dates: List[Date] = Field(..., description="List of dates.")


class DatesExtractor:
    """
    Extracts dates from a given passage and returns them in JSON format.
    """

    def __init__(self, model_name: str = "llama3.1", temperature: float = 0):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Extract dates from the given passage and return them in JSON format.",
                ),
                (
                    "user",
                    "Identify the dates from the passage and extract the day, month and year. Return the dates as a JSON object with a 'dates' key containing a list of date objects. Each date object should have 'day', 'month', and 'year' keys. Return the number of month, not name. \
                    Here is the text: {passage}",
                ),
            ]
        )

        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.output_parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def extract(self, passage: str) -> Dates:
        try:
            response = self.chain.invoke({"passage": passage})
            return Dates(**response)
        except Exception as e:
            logger.error(f"Error in extracting dates: {e}")
            return None


if __name__ == "__main__":
    extractor = DatesExtractor()
    passage1 = "Wikipedia, launched on January 15, 2001, is a free online encyclopedia created through collaborative editing. It provides information on diverse topics, contributed by volunteers worldwide. With millions of articles in multiple languages, Wikipedia has become a vital resource for knowledge sharing, accessible to anyone with internet access."

    passage2 = "Google, founded on Sept 4, 1998, by Larry Page and Sergey Brin, has grown into the world's leading search engine. Its services extend beyond search, offering tools like Gmail, Google Maps, and Google Drive. Every day, billions of users rely on Google for information, communication, and productivity."

    passage3 = "The first modern automobile was built on 29/01/1886 by Karl Benz, marking the birth of the car industry. Later, on 08/Jun/1948, Porsche unveiled its first car, the 356, revolutionizing the automotive world and setting the foundation for the sports car market we know today."

    for passage in [passage1, passage2, passage3]:
        result = extractor.extract(passage)
        print(result)


"""
Output:
dates=[Date(day=15, month=1, year=2001)]
dates=[Date(day=4, month=9, year=1998)]
dates=[Date(day=15, month=1, year=2001), Date(day=4, month=9, year=1998)]
"""
