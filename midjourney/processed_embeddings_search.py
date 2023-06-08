import os
import openai
import numpy as np
import pandas as pd
from typing import List
import math
import boto3
import dotenv


# loads .env file with your OPENAI_API_KEY
dotenv.load_dotenv()


def lambda_handler(event):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket="processed-embeddings", Key="docs.midjourney.com.csv")
    df = pd.read_csv(obj["Body"], index_col=0)

    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

    search_term = event["queryStringParameters"]["query"]

    df_return = answer_question(df, question=search_term)

    return {"statusCode": 200, "body": df_return}


def get_embeddings_for_text(input_term: str) -> List[List[float]]:
    """Get embedding of input text."""

    input_vector = openai.Embedding.create(
        input=input_term, model="text-embedding-ada-002"
    )
    input_vector_embeddings = input_vector["data"][0]["embedding"]

    return input_vector_embeddings


def create_context(
    question: str,
    df: pd.DataFrame,
    distance_metric: str = "cosine",
    max_len: int = 1800,
):
    """Create a context for a question by finding the most similar context from the dataframe."""

    # Get the embeddings for the question
    q_embeddings = get_embeddings_for_text(question)
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric=distance_metric
    )

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for _, row in df.sort_values("distances", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df: pd.DataFrame,
    model: str = "text-davinci-003",  # expensive
    question: str = "what is midjourney?",
    distance_metric: str = "cosine",
    max_len: int = 1800,
    max_tokens: int = 150,
    stop_sequence=None,
) -> str:
    """Answer a question based on the most similar context from the dataframe texts."""
    context = create_context(
        question,
        df,
        distance_metric,
        max_len,
    )

    try:
        prompt = f"""
        Answer the question based on the context below, and if the question can't be answered based on the context, say "I don't know".
        
        Question: {question}
        =========
        Context: {context}
        
        Answer:
        """
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


def cosine_distance(a: List[float], b: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return 1.0 - dot_product / (magnitude_a * magnitude_b)


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": cosine_distance,
        "L1": None,  # To be implemented
        "L2": None,  # To be implemented
        "Linf": None,  # To be implemented
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


# curl -G - data-urlencode "query=YOUR QUERY"
# "https://YOUR_FUNCTION_URL_ID.lambda-url.us-east-1.on.aws/"
