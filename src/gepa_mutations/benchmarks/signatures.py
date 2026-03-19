"""DSPy signatures used by benchmark evaluators."""

import dspy


class MathSolverSignature(dspy.Signature):
    """Solve a math problem step by step."""

    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")
