import re
from openai import OpenAI


def numerize_label(label):
    """
    Converts label description into numbers for easy comparison. 

    Parameters:
    label (str): The label description. 

    Returns:
    str: The corresponding numerized label. 
    """
    if "output_1 is much better than output_2" in label:
        return 1
    elif "output_1 is slightly better than output_2" in label:
        return 2
    elif "output_1 is about the same as output_2" in label:
        return 3
    elif "output_1 is slightly worse than output_2" in label:
        return 4
    elif "output_1 is much worse than output_2" in label:
        return 5
    else:
        raise ValueError("Invalid label provided.")

def downscale_label(label):
    """
    Converts a 5-point scale label to a 3-point scale.

    Parameters:
    label (str): The label on the 5-point scale. 

    Returns:
    str: The corresponding response on the 3-point scale.
    """
    if "better" in label:
        return "output_1 is much better than output_2"
    elif "same" in label:
        return label # leave the same
    elif "worse" in label:
        return "output_1 is much worse than output_2"
    else:
        raise ValueError("Invalid label provided.")

def invert_label(label):
    """
    Inverts a 5-point scale label. Useful for positional tests. 
    
    E.g. invert_label("output_1 is much better than output_2")
         > "output_1 is much worse than output_2"

    Parameters:
    label (str): The label on the 5-point scale. 

    Returns:
    str: The corresponding inverted response.
    """
    if "better" in label:
        return label.replace("better", "worse")
    elif "same" in label:
        return label # no change required. 
    elif "worse" in label:
        return label.replace("worse", "better")
    else:
        raise ValueError("Invalid label provided.")

# TODO: This shouldn't repeated in utils.py.
def read_text_file(file_path):
    """
    Reads a text file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file to be read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file cannot be found at the specified path.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' does not exist.")
        raise
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def add_prefix_to_keys(original_dict, prefix):
    # Create a new dictionary with the prefix added to each key
    new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
    return new_dict

def extract_response_info(sentence):
    # Enhanced regular expression with corrected spacing and flexible matching
    pattern = re.compile(
        r"(response [ab]).*(much better|a little better|better|similar|a little worse|worse|much worse).*?(response [ab])",
        re.IGNORECASE
    )

    # Search for patterns in the sentence
    match = pattern.search(sentence)

    if match:
        response_first = match.group(1).lower()
        comparison = match.group(2).lower()
        if "much" in sentence:
          comparison = "much " + comparison
        elif "a little" in sentence:
          comparison = "a little " + comparison
        response_second = match.group(3).lower()

        # Ensure "response a" is always discussed first in the output
        if response_first.endswith("b"):
            # Reverse the comparison if "response b" is mentioned first
            reverse_comparison_map = {
                "much better": "much worse",
                "a little better": "a little worse",
                "better": "worse",
                "similar": "similar",
                "a little worse": "a little better",
                "worse": "better",
                "much worse": "much better"
            }
            adjusted_comparison = reverse_comparison_map[comparison]
            return ["response a", adjusted_comparison]
        else:
            return ["response a", comparison]
    else:
        return ["", ""]
    
def query_openai_with_history(initial_prompt, follow_up_prompt, model = "gpt-4o"):
    client = OpenAI()

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt}
    ]
    )

    first_response = completion.choices[0].message
    
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt},
        {'role': "assistant", "content": first_response.content},
        {"role": "user", "content": follow_up_prompt},
    ]
    )

    second_response = completion.choices[0].message
    
    return first_response, second_response