from openai import APIConnectionError, OpenAI, RateLimitError
from loguru import logger
import jsonlines

def attempt_api_call(model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    for attempt in range(max_retries):
        try:
            # Ollama
            if model_name == "Ollama":
                model = Ollama(model="llama3.1:70b-instruct-fp16",  base_url='http://147.46.215.218:3000/ollama', headers={"Authorization": "Bearer sk-34bee5358a1f49f9ade2174545324f95", "Content-Type": "application/json"},)
                result = model.invoke(messages)
                return result
            # gpt-4
            elif model_name == "gpt-4":
                openai_keys = get_openai_api_key()
                openai_client = OpenAI(api_key=openai_keys)
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                return response.choices[0].message.content

        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)

def evaluate_predictions(ag_output, segments, top100_segments, model_name):
    score = 0
    len_ag_output = 0
    with jsonlines.open(ag_output, "r") as file:
        for idx, line in enumerate(tqdm(file, desc="Evaluating Predictions")):
            question = line["topic"]
            answer = line["answer"]
            citations = list(set([ind for a in line["answer"] for ind in a["citations"]]))
            ref_docid = [line["references"][i] for i in citations]
            references = []
            with jsonlines.open(segments, "r") as seg_file:
                for line_seg in seg_file:
                    docid = line_seg["docid"]
                    if docid in ref_docid:
                        references.append(line_seg)

            system_prompt = '''
            Keep in mind the following Guidelines when evaluating the answer and references:
            #################### Guidelines:
                • Citation requirement identification : Whether each answer sentence is identified well for citation requirement. It is well identified if sentence requires a citation and a citation is provided correctly, or if the sentence is a generic sentence requiring no citation and citation is not provided.
                • Citation Supportiveness : For answer sentences containing citations, how well each sentence is supported by its associated cited segments.  
                • Fluency : Whether each answer sentence is fluent and cohesive, based on coherence, grammar and readability.
                • Nugget coverage : How many nuggets from the list are present within the answer. Whether the nugget is not supported by the answer, partially supported, or fully supported.

            ##################### Instructions: Question, answer sentences, and references will be given. Please read them carefully along with the Guidelines for how to evaluate an answer’s quality. Then:
                • Create "Nuggets" list by using Question References and question given. Using the relevant references for each input question, a nugget creation (or nuggetization) is to generate multiple few-word to sentence-long nuggets containing factual information relevant to the input question. The “Nuggets” list is constructed for every question selected. These nuggets span all crucial facts that an RAG answer should cover. These “Nuggets” also include a binary score of vitality (‘okay’ or ‘vital’). Vitality is a binary classification that indicates how crucial or significant a nugget is for the given question. In your description, nuggets are assigned a vitality score of either ‘okay’ or ‘vital'. 'Vital' means the nugget is considered crucial or highly relevant to the question. 'Okay' implies that the nugget is relevant but not critical. 
                • Score overall answer on 1-100, where 100 is a perfect answer that aligns with the Guidelines. Each answer sentence’s citation is contained within the “citations” key of its corresponding dictionary. These citations are represented as 0-based indices that refer to specific entries in references list.
            When you are finished, return your response with only number.
            '''
            # Ollama
            if model_name == "Ollama":
                message = f"{system_prompt}\nQuestion: {question}\nAnswer sentences: {answer}\nAnswer References: {references}\nQuestion References {references_q}"

            # gpt-4
            elif model_name == "gpt-4":
                message = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nAnswer sentences: {answer}\nAnswer References: {references}\nQuestion References {references_q}",
                    },
                ]


            response = attempt_api_call(model_name, message)
            if response:
                try:
                    int_response = int(response)
                    score += int_response
                    len_ag_output += 1
                    log_response(message, response)
                except ValueError:
                    pass
                
        results = {
            "Score" : score / len_ag_output
        }
        logger.info(results)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-4', type=str)
    parser.add_argument('--ag_output', default='ag_output_trec_rag_2024.jsonl', type=str)
    parser.add_argument('--segment', default='msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.raggy-dev', type=str)
    parser.add_argument('--top100_segment', default='top100_segment', type=str)

    args = parser.parse_args()
    evaluate_predictions(args.ag_output, args.segment, args.top100_segment, args.model)