import random

def generate_expression(variables):
    # Define the possible operators for comparisons
    operators = ["<", ">", "==", "!="]

    # Define the possible boolean operators
    bool_operators = ["and", "or"]

    def generate_comparison(variables):
        if random.random() < 0.67:
            # Generate a comparison between two variables
            var1, var2 = random.sample(variables, 2)
            return f"{var1} {random.choice(operators)} {var2}"
        else:
            # Generate a comparison between a variable and a random value
            var = random.choice(variables)
            value = random.randint(0, 100)
            return f"{var} {random.choice(operators)} {value}"

    def generate_recursive(variables, depth=0):
        if depth >= 3 or random.random() < 0.3:
            return generate_comparison(variables)

        left = generate_recursive(variables, depth + 1)
        right = generate_recursive(variables, depth + 1)
        return f"({left} {random.choice(bool_operators)} {right})"

    return generate_recursive(variables)

# Example usage

with open("expression-examples.txt", "w") as f:
    for i in range(10000):
        expression = generate_expression(["a", "b", "c", "d"])
        print(expression)
        f.write(f"lambda a, b, c, d: {expression}\n" )



# Convert the expression to a lambda function
func = eval(f"lambda a, b, c, d: {expression}")
print(func(1, 2, 3, 4))
