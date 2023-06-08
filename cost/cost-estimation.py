# import tiktoken

# enc = tiktoken.encoding_for_model("text-embedding-ada-002")

# total_word_count = sum(len(doc.page_content.split()) for doc in data_split)
# total_token_count = sum(len(enc.encode(doc.page_content)) for doc in data_split)

# print(f"\nTotal word count: {total_word_count}")
# print(f"\nEstimated tokens: {total_token_count}")
# print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")


# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import OpenAI

# llm = OpenAI(temperature=0)
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# with get_openai_callback() as cb:
#     response = agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
#     print(f"Total Tokens: {cb.total_tokens}")
#     print(f"Prompt Tokens: {cb.prompt_tokens}")
#     print(f"Completion Tokens: {cb.completion_tokens}")
#     print(f"Total Cost (USD): ${cb.total_cost}")
