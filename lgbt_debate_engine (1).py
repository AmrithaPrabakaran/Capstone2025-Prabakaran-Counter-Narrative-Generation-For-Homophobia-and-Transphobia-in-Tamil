import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt defining the assistant's role and process
SYSTEM_PROMPT = """
You are a debate-driven assistant focused on deconstructing and refuting homophobic and transphobic rhetoric through a multistep process that includes persona generation, team formation, discussion planning, collaborative debate, and counter narrative formulation. Given an input text containing anti-LGBT+ sentiment, your role is to guide users through five distinct stages:

1. Persona Creation: Generate 6–10 diverse LGBT+ agents with unique identities and specific counter-narrative strategies to oppose the input text.
2. Team Selection: From the pool of personas, select a team of three agents with varied perspectives, giving reasons for their selection based on strategic alignment and diversity.
3. Debate Discussion Plan: Simulate a structured, multi-turn debate between the three selected agents (opposing the text) and a critic (supporting the text). The goal is to collaboratively construct a logical and persuasive plan.
4. Plan Distillation: Summarize the discussion into an abstract, structured plan with up to three main points, each supported by sub-points and optional acknowledgments.
5. Counter Narrative Generation: Based on the finalized plan, generate a concise yet powerful counter-narrative in one or two lines.

You must maintain fairness, clarity, and rigor throughout each step, encouraging logical reasoning, intersectional representation, and respectful discourse. Clarify ambiguity when needed, and keep your tone professional, constructive, and empowering.
"""

# Function to interact with GPT

def query_gpt(user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message['content']

# Full Process

def run_debate_engine(input_text):
    print("--- Step 1: Persona Creation ---")
    personas_prompt = f"Input: '{input_text}'\nGenerate 6–10 diverse LGBT+ personas with unique identities and counter-narrative strategies."
    personas = query_gpt(personas_prompt)
    print(personas)

    print("\n--- Step 2: Team Selection ---")
    team_prompt = f"Input: '{input_text}'\nBased on these personas:\n{personas}\nSelect a team of 3 with varied perspectives, explaining your choices."
    team = query_gpt(team_prompt)
    print(team)

    print("\n--- Step 3: Debate Discussion Plan ---")
    debate_prompt = f"Input: '{input_text}'\nSimulate a structured, multi-turn debate between the 3 selected agents and a critic."
    debate = query_gpt(debate_prompt)
    print(debate)

    print("\n--- Step 4: Plan Distillation ---")
    plan_prompt = f"Summarize the above debate into a structured plan with 3 main points and supporting details."
    plan = query_gpt(plan_prompt)
    print(plan)

    print("\n--- Step 5: Counter Narrative Generation ---")
    narrative_prompt = f"Based on this plan:\n{plan}\nGenerate a concise, powerful 1–2 sentence counter-narrative."
    counter_narrative = query_gpt(narrative_prompt)
    print(counter_narrative)

# Example usage
if __name__ == "__main__":
    run_debate_engine("LGBT+ rights are a threat to traditional values.")
