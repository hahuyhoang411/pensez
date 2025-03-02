from transformers import AutoTokenizer
import re
import fasttext 
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

def ends_with_question_mark(example):
    """
    Check if the instruction ends with a question mark.
    Returns True if it does, False otherwise.
    """
    # Check if 'instruction' field exists and has a value
    if not example.get('instruction'):
        # If not available, check the human message in conversations
        conversations = example['conversations']
        human_messages = [conv['value'] for conv in conversations if conv['from'] == 'human']
        if not human_messages:
            return False
        instruction = human_messages[0]
    else:
        instruction = example['instruction']
    
    # Strip whitespace and check if it ends with a question mark
    return instruction.strip().endswith('?')


def count_tokens_combined(example, tokenizer=None, model_path=None, use_concatenated_text=False):
    """
    Count tokens in an example. Can count tokens in messages or in concatenated text fields.
    
    Parameters:
    - example: The data example to process.
    - tokenizer: The tokenizer to use (required if use_concatenated_text is False).
    - model_path: The model path to load the tokenizer from (required if use_concatenated_text is True).
    - use_concatenated_text: Boolean flag to determine which fields to tokenize.
    
    Returns:
    - The example with an added 'token_count' field.
    """
    if use_concatenated_text:
        if model_path is None:
            raise ValueError("model_path must be provided when use_concatenated_text is True.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        text = example["question"] + example["deepseek_thinking_trajectory"] + example["deepseek_attempt"]
        token_count = len(tokenizer(text).input_ids)
    else:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_concatenated_text is False.")
        total_tokens = 0
        for message in example['messages']:
            tokens = tokenizer(message['content'], return_tensors="pt", truncation=False)
            total_tokens += len(tokens['input_ids'][0])
        token_count = total_tokens
    
    example['token_count'] = token_count
    return example


def is_fr_input(example):
    """
    Check if the user input is in French using fasttext.
    Returns True if the input is in French, False otherwise.
    """
    # Get the user input
    conversations = example['messages']
    user_inputs = [conv['content'] for conv in conversations if conv['role'] == 'user']
    
    if not user_inputs:
        return False
    
    # Use the first 500 characters for language detection
    text_to_check = user_inputs[0]
    
    # Remove newlines and replace with spaces
    text_to_check = text_to_check.replace('\n', ' ')
    
    # Predict language
    predictions = model.predict(text_to_check)
    predicted_lang = predictions[0][0]
    confidence = predictions[1][0]
    
    # Check if it's French with high confidence
    is_french = predicted_lang == '__label__fra_Latn' and confidence > 0.99
    
    return is_french

def is_fr_response(example):
    """
    Check if the assistant response is in French using fasttext.
    Returns True if the response is in French, False otherwise.
    """
    # Get the GPT response
    conversations = example['messages']
    gpt_responses = [conv['content'] for conv in conversations if conv['role'] == 'assistant']
    
    if not gpt_responses:
        return False
    
    # Use the first 500 characters for language detection
    text_to_check = gpt_responses[0]
    
    # Remove newlines and replace with spaces
    text_to_check = text_to_check.replace('\n', ' ')
    
    # Predict language
    predictions = model.predict(text_to_check)
    predicted_lang = predictions[0][0]
    confidence = predictions[1][0]
    
    # Check if it's French with high confidence
    is_french = predicted_lang == '__label__fra_Latn' and confidence > 0.99
    
    return is_french

def is_en_response(example):
    """
    Check if the assistant response is in English using fasttext.
    Returns True if the response is in English, False otherwise.
    """
    # Get the assistant response
    conversations = example['messages']
    assistant_responses = [conv['content'] for conv in conversations if conv['role'] == 'assistant']
    
    if not assistant_responses:
        return False
    
    # Use the first 500 characters for language detection
    text_to_check = assistant_responses[0]
    
    # Remove newlines and replace with spaces
    text_to_check = text_to_check.replace('\n', ' ')
    
    # Predict language
    predictions = model.predict(text_to_check)
    predicted_lang = predictions[0][0]
    confidence = predictions[1][0]
    
    # Check if it's English with high confidence
    is_english = predicted_lang == '__label__eng_Latn' and confidence > 0.95
    
    return is_english


def is_en_input(example):
    """
    Check if the response is actually in French using fasttext.
    Returns True if the response is in French, False otherwise.
    """
    # Get the GPT response
    conversations = example['messages']
    gpt_responses = [conv['content'] for conv in conversations if conv['role'] == 'user']
    
    if not gpt_responses:
        return False
    
    # Use the first 500 characters for language detection
    text_to_check = gpt_responses[0]
    
    # Remove newlines and replace with spaces
    text_to_check = text_to_check.replace('\n', ' ')
    
    # Predict language
    predictions = model.predict(text_to_check)
    predicted_lang = predictions[0][0]
    confidence = predictions[1][0]
    
    # Check if it's French with high confidence
    is_eng = predicted_lang == '__label__eng_Latn' and confidence > 0.99
    
    return is_eng

def is_en_reasoning(example):
    """
    Check if the response is actually in French using fasttext.
    Returns True if the response is in French, False otherwise.
    """
    # Get the GPT response
    gpt_responses = example['reasoning']
    
    if not gpt_responses:
        return False
    
    # Use the first 500 characters for language detection
    text_to_check = gpt_responses[0]
    
    # Remove newlines and replace with spaces
    text_to_check = text_to_check.replace('\n', ' ')
    
    # Predict language
    predictions = model.predict(text_to_check)
    predicted_lang = predictions[0][0]
    confidence = predictions[1][0]
    
    # Check if it's French with high confidence
    is_eng = predicted_lang == '__label__eng_Latn' and confidence > 0.95
    
    return is_eng
    
def extract_answer(text):
    """
    Extract answer using multiple strategies in order of precedence:
    1. \boxed{...} format
    2. Answer: format
    3. Raw text as fallback
    """
    if text is None:
        return None
        
    # Strategy 1: Extract from \boxed{...}
    boxed_pattern = r'\\boxed{([^}]*)}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
        
    # Strategy 2: Extract from "Answer: ..." format
    answer_pattern = r'(?i)answer:\s*(.+?)(?:\n|$)'
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        return answer_match.group(1).strip()
        
    # Strategy 3: Use the raw text as fallback
    return text.strip()

def normalize_answer_format(answer):
    """
    Normalize different answer formats to a standard format 'A, B, C, D'
    Handles various formats and separators including special characters.
    """
    if answer is None:
        return None
        
    # Convert to uppercase and strip whitespace
    answer = answer.strip().upper()
    
    # If single character answer, return it
    if len(answer) == 1 and answer.isalpha():
        return answer
    
    # Step 1: Replace all non-alphanumeric characters with comma
    standardized = re.sub(r'[^A-Z0-9]+', ',', answer)
    
    # Step 2: Handle case where options are just concatenated (e.g., 'ABCD')
    if ',' not in standardized:
        if all(c.isalpha() for c in standardized):
            standardized = ','.join(list(standardized))
    
    # Step 3: Clean up the options
    options = []
    for opt in standardized.split(','):
        opt = opt.strip()
        if opt and (
            (len(opt) == 1 and opt.isalpha()) or  # Single letter
            opt.isdigit() or                      # Number
            (len(opt) <= 2 and opt.isalnum())     # Alphanumeric up to 2 chars
        ):
            options.append(opt)
    
    # Step 4: Remove duplicates while preserving order
    seen = set()
    options = [x for x in options if not (x in seen or seen.add(x))]
    
    # Step 5: Join with standard separator
    return ', '.join(options) if options else None

def compare_answer(extracted, correct):
    """
    Compare extracted answer with correct answer.
    Both answers are normalized before comparison.
    """
    if extracted is None or correct is None:
        return False
    
    # First extract the answer using our new extraction logic
    extracted_clean = extract_answer(extracted)
    correct_clean = extract_answer(correct)
    
    # Then normalize both answers
    normalized_extracted = normalize_answer_format(extracted_clean)
    normalized_correct = normalize_answer_format(correct_clean)
    
    # Handle None cases after normalization
    if normalized_extracted is None or normalized_correct is None:
        return False
    
    return normalized_extracted == normalized_correct

def process_dataset(dataset):
    """
    Process the dataset and add correctness column for each generation
    """
    def process_example(example):
        extracted_answer = extract_answer(example['deepseek_attempt'])
        is_correct = compare_answer(extracted_answer, example['solution'])
        example['correctness'] = is_correct
        return example
        
    processed_dataset = dataset.map(process_example, num_proc=32)
    return processed_dataset

def create_conversation_format(entry):
    """
    Convert a dataset entry into the desired conversation format.
    
    Args:
        entry (dict): Dictionary containing 'question' and 'solution' keys
        
    Returns:
        list: List of conversation dictionaries
    """
    solution = entry['solution']
    
    # Split into sentences (handling both '. ' and '.\n' cases)
    sentences = re.split(r'(?<=\.)[\s\n]+', solution)
    
    # Remove empty sentences and trim each sentence
    filtered_sentences = [s.strip() for s in sentences if s.strip()]
    
    # Get the last sentence and everything before it
    last_sentence = filtered_sentences[-1]
    reasoning = '. '.join(filtered_sentences[:-1])
    
    # Create the conversation structure
    conversations = [
        {
            "from": "human",
            "value": entry['question']
        },
        {
            "from": "gpt",
            "value": f"<think>\n{reasoning}\n</think>\n\n{last_sentence}"
        }
    ]
    
    return conversations