"""
input: formula strings
output: the answer of the mathematical formula
"""

import re
from operator import truediv, mul, add, sub
import wolframalpha


def calculator(query: str):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv,
    }

    query = re.sub(r'\s+', '', query)

    if query.isdigit():
        return float(query)

    for op in operators.keys():
        left, operator, right = query.partition(op)
        if operator in operators:
            return round(operators[operator](calculator(left), calculator(right)), 2)


def WolframAlphaCalculator(input_query: str):
    try:
        wolfram_alpha_appid = "xxxx"
        wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
        res = wolfram_client.query(input_query)
        answer = next(res.results).text

    except Exception:
        raise Exception(
            "Invalid input query for Calculator. Please check the input query or use other functions to do the computation."
        )

    return answer


if __name__ == "__main__":
    query = "max(37.97,76.1)"
    print(WolframAlphaCalculator(query))