# Starter file for Cursor AI
print("Hello, Cursor!")
# initial commit

def reverse_string(input_string: str) -> str:
    """
    Return a new string that is the reverse of `input_string`.
    """
    # 1) Validate the argument is a string
    if not isinstance(input_string, str):
        raise TypeError(
            f"input_string must be a str, got {type(input_string).__name__}"
        )

    # 2) Use slicing with a step of -1 to reverse characters
    reversed_string = input_string[::-1]

    # 3) Return the reversed result
    return reversed_string

if __name__ == "__main__":
    # Example usage
    example_value = "Cursor"
    print(reverse_string(example_value))
