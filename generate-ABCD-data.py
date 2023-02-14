import pandas as pd
import math, random 
from numpy.random import randint

df = pd.DataFrame(columns=['A','B','C','D','R'])

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


def gtAdd (v):
 #  return v[0] % 2 == 1
   return   v[3] == v[0] + v[1] / (v[2]+20)
   return   math.sin(v[1])  == v[3]

def gtAndOr (v):
   return  ((v[0] < 20) or (v[1] > 30)) and ((v[2] <45) or (v[3] < 30)) 

def gtEven (v):
   return  v[0]%2 == 1

expression = generate_expression(["a", "b", "c", "d"])
print(expression)

# Convert the expression to a lambda function
func = eval(f"lambda a, b, c, d: {expression}")

print (df)
for i in range(7000):
   v = list(randint(100, size=4))
   #df.loc[i] = v + [func(v[0],v[1],v[2],v[3])]
   #df.loc[i] = v + [random.random() > 0.5]
   df.loc[i] = v + [func(v[0],v[1],v[2],v[3])]  
   if False:
    if i%2:
        v[3] = math.sin((v[1]+v[2])/20)
        df.loc[i] = v + [True]
    else:
        v[3] = math.sin((v[1]+v[2])/20)+0.1
        df.loc[i] = v + [False]
            
print (df)
df.to_csv('testdata.csv', index=False)