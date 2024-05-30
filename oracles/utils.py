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

def add_prefix_to_keys(original_dict, prefix):
    # Create a new dictionary with the prefix added to each key
    new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
    return new_dict