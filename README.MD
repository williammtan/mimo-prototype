
Parameters:

1. Model: size vs. price
2. Use reasoning or not
3. Weights:
 - Do you want to weight each scores?
 - If so, who will decide the weights? Another LLM based on job description?
4. Prompt engineering
5. Metrics? How to evaluate the performance?
6. Few-shot prompting
 - Give LLM examples to follow
 - Examples can be stored and retrieved from the vector DB
 - + Batch prompting: we batch multiple resumes into one prompt, reducing the repeated tokens for instruction and examples