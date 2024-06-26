
#Prompt Engineering Langchain
import os
os.environ["OPENAI_API_KEY"] = "Paste your API Key here"
from langchain import PromptTemplate

demo_template='''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt=PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
    )

prompt.format(financial_concept='income tax')
'I want you to act as a acting financial advisor for people.\nIn an easy way, explain the basics of income tax.'
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm=OpenAI(temperature=0.7)
chain1=LLMChain(llm=llm,prompt=prompt)
chain1.run('GDP')
#Output:- "\n\nGDP stands for Gross Domestic Product and is an economic measure of a country's total economic output. It is a measure of the value of all goods and services produced within a country's borders in a given period of time, typically a year. It includes consumer spending, government spending, investments, and exports minus imports. This measure is important because it can be used to compare the economic performance of different countries and to assess the overall health of an economy."
## Language Translation

from langchain import PromptTemplate

template='''In an easy way translate the following sentence '{sentence}' into {target_language}'''
language_prompt = PromptTemplate(
    input_variables=["sentence",'target_language'],
    template=template,
)
language_prompt.format(sentence="How are you",target_language='hindi')
"In an easy way translate the following sentence 'How are you' into hindi"
chain2=LLMChain(llm=llm,prompt=language_prompt)

chain2({'sentence':"Hello How are you",'target_language':'hindi'})
"""{'sentence': 'Hello How are you',
 'target_language': 'hindi',
 'text': '\n\nनमस्ते आप कैसे हैं?'}"""
from langchain import PromptTemplate, FewShotPromptTemplate

# First, create the list of few shot examples.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# Next, we specify the template to format the examples we have provided.
# We use the `PromptTemplate` class for this.
example_formatter_template = """Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)
# Finally, we create the `FewShotPromptTemplate` object.
few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give the antonym of every input\n",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym: ",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n",
)
print(few_shot_prompt.format(input='big'))
"""Give the antonym of every input

Word: happy
Antonym: sad

Word: tall
Antonym: short

Word: big
Antonym: """
chain=LLMChain(llm=llm,prompt=few_shot_prompt)
chain({'input':"big"})
{'input': 'big', 'text': ' small'}
