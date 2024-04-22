#%%
import numpy as np
from openai import OpenAI
import json
from openai_utils import OPENAI_API_KEY


#LLM_MODEL = 'text-davinci-003'
LLM_MODEL = 'gpt-3.5-turbo-instruct'


def live_query():
    client = OpenAI(api_key=OPENAI_API_KEY)
    while True:
        prompt = input()
        completion = client.completions.create(model=LLM_MODEL, prompt=[prompt], temperature=0., max_tokens=300)
        assert(len(completion.choices) == 1)
        print(completion.choices[0].text)


if __name__ == '__main__':
    live_query()
