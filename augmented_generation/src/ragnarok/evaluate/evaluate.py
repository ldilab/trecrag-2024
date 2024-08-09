from openai import APIConnectionError, OpenAI, RateLimitError
from loguru import logger
import jsonlines
from tqdm.auto import tqdm
import argparse
import vllm
import os
import re
import json
from collections import defaultdict

VLLM_TENSOR_PARALLEL_SIZE = 2
VLLM_GPU_MEMORY_UTILIZATION = 0.85
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def attempt_api_call(agent, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    for attempt in range(max_retries):
        try:
            # llama
            if "llama" in model_name:
                tokenizer = agent.get_tokenizer()
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                responses = agent.generate(
                    prompt,
                    vllm.SamplingParams(
                        n=1,  # Number of output sequences to return for each prompt.
                        temperature=0.1,  # Randomness of the sampling
                        skip_special_tokens=True,  # Whether to skip special tokens in the output.
                        max_tokens= 1000,  # Maximum number of tokens to generate per output sequence.
                    ),
                    use_tqdm=False
                )
                result = [response.outputs[0].text for response in responses][0]
                return result

            # gpt
            elif "gpt" in model_name:
                response = agent.chat.completions.create(
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

def nuggetize(agent, relevant_segments, model_name):
    """Return nuggets dictionary with query(key) and nuggets list(value)."""
    nuggets_dict = defaultdict(list)
    nuggetize_prompt = '''
        By using the provided "Question" and "Question Reference," create "Nuggets" if the reference is relevant to the question. Generate a list of nuggets where each nugget is accompanied by a vitality score.

        Follow these instructions exactly:
        1. Each nugget should be a brief statement or fact relevant to the question.
        2. Each nugget should include a binary vitality score: 'okay' or 'vital'.
        3. Return the nuggets in the format specified below:
        - Each nugget should be on a new line.
        - The vitality score should be included within parentheses right after the nugget text.

        For example:
        Landlord is liable if tenant is injured due to negligence (vital)
        Landlord is responsible for property maintenance (okay)

        Please provide the nuggets in the exact format specified above. Do not include any additional explanation or text (ex. "Here are the nuggets:").
        '''
    with jsonlines.open(relevant_segments, "r") as file:
        for line in tqdm(file, desc="Nuggetizing for queries"):
            question = line["query"]["text"]
            references_q = line["candidates"]
            for ref in tqdm(references_q, desc="Nuggetizing segments"):
                message = [
                            {"role": "system", "content": nuggetize_prompt},
                            {
                                "role": "user",
                                "content": f"Question: {question}\nQuestion Reference: {ref}\n",
                            },
                        ]
                response = attempt_api_call(agent, model_name, message)
                # print("Response: ", response)
                try:
                    nuggets = [line.strip() for line in response.strip().split("\n") if line.strip()]
                    nuggets_dict[question].extend(nuggets)
                except:
                    continue

    return nuggets_dict


def nugget_coverage_score(agent, query, answer_sentences, nuggets_list, model_name):
    """Estimate nugget coverage score for answer sentences with LLM."""
    system_prompt = '''
        You are a nugget coverage score calculator. Given the following nugget and a list of answer sentences, calculate the "Nugget Coverage Score" and return only the numerical score. The coverage score should be a percentage (0 to 100), rounded to two decimal places.

        **Example Calculation**:

        Nugget: "Landlord is liable if tenant is injured due to negligence (vital)"
        Answer Sentences: ["Landlord must address hazardous conditions.", "Landlords can be held liable if their negligence leads to tenant injury.", "Landlords need to maintain their property.", "This sentence does not support the nugget."]
        Coverage Score Calculation:
        - Score sentences based on how well they support the nugget (100 for fully supported, 50 for partially supported, 0 for not supported).
        - Total score = sum of individual scores.
        - Coverage Score = \(\left(\frac{\text{Total Score}}{\text{Number of Sentences} \times 100}\right) \times 100\).

        **Return Only the Coverage Score**: Provide the coverage score as a number with no additional text or explanation.
        '''
    avg_nugget_coverage = 0
    len_nuggets = 0
    # First 10 nuggets
    for nugget in tqdm(nuggets_list[:10], desc="Estimating nugget coverage score"):
        message = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Nugget: {nugget}\nAnswer Sentences: {answer_sentences}\n",
                    },
                ]
        response = attempt_api_call(agent, model_name, message)
        print("Score: ", response)
        if response:
            try:
                int_response = int(response)
                avg_nugget_coverage += int_response
                len_nuggets += 1
            except ValueError:
                pass

    result = avg_nugget_coverage / len_nuggets   
    print("Average nugget score: ", result)  
    return result
        
def save_nuggets_dict(file_path, agent, top100_segments, model_name):
    """Nuggetize and save nuggets dictionary if not exists."""
    if not os.path.exists(file_path):
        # Nuggetization
        nuggets_dict = nuggetize(agent, top100_segments, model_name)
        with open(file_path, "w") as f:
            json.dump(nuggets_dict, f)
        print(f"File '{file_path}' created and data saved.")
    else:
        print(f"File '{file_path}' already exists. Skipping save.")
        # load nuggets_dict
        with open(file_path, 'r') as file:
            nuggets_dict = json.load(file)
            print(f"File '{file_path}' data loaded.")
    
    return nuggets_dict

def save_nuggets_score(file_path, agent, ag_output, nuggets_dict, model_name):
    """Get nugget coverage score and save if not exists."""
    if not os.path.exists(file_path):
        # Get nugget coverage score
        with jsonlines.open(ag_output, "r") as file:
            for line in tqdm(file, desc="Get nugget coverage score"):
                question = line["topic"]
                answer_sentences = [a["text"] for a in line["answer"]]
                score = nugget_coverage_score(agent, question, answer_sentences, nuggets_dict[question], model_name)
                ncs_dict[question] = score
        with open(file_path, "w") as f:
            json.dump(ncs_dict, f)
        print(f"File '{file_path}' created and data saved.")
    else:
        print(f"File '{file_path}' already exists. Skipping save.")
        # load nugget_coverage_score
        with open(file_path, 'r') as file:
            ncs_dict = json.load(file)
            print(f"File '{file_path}' data loaded.")
    
    return ncs_dict

def evaluate_predictions(ag_output, top100_segments, model_name):
    """Evaluate answer sentences based on support, fluency, nugget coverage."""
    overall_score = 0
    len_ag_output = 0
    ncs_dict = defaultdict(int)

    # llama agent
    if "llama" in model_name:
        agent = vllm.LLM(
            model_name,
            download_dir=os.getenv("HF_HOME"),
            enforce_eager=True,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
        )
    # gpt agent
    elif "gpt" in model_name:
        openai_keys = get_openai_api_key()
        agent = OpenAI(api_key=openai_keys)

    # Make nuggets_dir
    nuggets_dir = "nuggets"
    os.makedirs(nuggets_dir, exist_ok=True)
    
    # Save nuggets_dict
    filepath_nuggets = os.path.join(nuggets_dir, "nuggets_dict")
    nuggets_dict = save_nuggets_dict(filepath_nuggets, agent, top100_segments, model_name)
    
    # Save nugget coverage score
    filepath_nuggets_score = os.path.join(nuggets_dir, "nuggets_score")
    ncs_dict = save_nuggets_score(filepath_nuggets_score, agent, ag_output, nuggets_dict, model_name)

    # Evaluate answer sentences
    with jsonlines.open(ag_output, "r") as file:
        with jsonlines.open(top100_segments, "r") as seg_file:
            for line, line_seg in tqdm(zip(file, seg_file), desc="Evaluating Predictions"):
                question = line["topic"]
                answer = line["answer"]
                citations = list(set([ind for a in line["answer"] for ind in a["citations"]]))
                ref_docid = []
                for i in citations:
                    if i < len(line["references"]):
                        ref_docid.append(line["references"][i])
                references = []
                for candidate in line_seg["candidates"]:
                    docid = candidate["docid"]
                    if docid in ref_docid:
                        references.append(line_seg)

                # nugget coverage score
                ncs = ncs_dict[question]

                system_prompt = '''
                Keep in mind the following Guidelines when evaluating the answer and references:
                #################### Guidelines:
                    • Citation requirement identification : Whether each answer sentence is identified well for citation requirement. It is well identified if sentence requires a citation and a citation is provided correctly, or if the sentence is a generic sentence requiring no citation and citation is not provided.
                    • Citation Supportiveness : For answer sentences containing citations, how well each sentence is supported by its associated cited segments.  
                    • Fluency : Whether each answer sentence is fluent and cohesive, based on coherence, grammar and readability.
                    • Nugget coverage score: How many nuggets from the list are present within the answer. Whether the nugget is not supported by the answer, partially supported, or fully supported. Score is 0-100.

                ##################### Instructions: Question, answer sentences, references, and nugget coverage score will be given. Please read them carefully along with the Guidelines for how to evaluate an answer’s quality. Then:
                    • Score overall answer on 1-100, where 100 is a perfect answer that aligns with the Guidelines. Each answer sentence’s citation is contained within the “citations” key of its corresponding dictionary. These citations are represented as 0-based indices that refer to specific entries in references list.
                When you are finished, return your response with only number. There is no need to explain the reasoning behind your answers.
                '''

                message = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Question: {question}\nAnswer sentences: {answer}\nReferences: {references}\nNugget coverage score {ncs}\n",
                        },
                    ]

                response = attempt_api_call(agent, model_name, message)
                print("Overall score: ", response)
                if response:
                    try:
                        int_response = int(response)
                        overall_score += int_response
                        len_ag_output += 1
                        log_response(message, response)
                    except ValueError:
                        pass
                
        results = {
            "Score" : overall_score / len_ag_output
        }
        logger.info(results)
        return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-4o-mini', help='evaluation model', type=str)
    parser.add_argument('--ag_output_path', default='ag_output_trec_rag_2024.jsonl', help='RAG output for evaluation', type=str)
    parser.add_argument('--top100_segments_path', default='../../../retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.raggy-dev_top100_sample.jsonl', help='Top-100 corpus segments to use', type=str)

    args = parser.parse_args()
    evaluate_predictions(args.ag_output_path, args.top100_segments_path, args.model)