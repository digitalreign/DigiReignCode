# 20191002 BeginningPython Chapter 5
# Conditional, Loops, and Some Other Statements

# 20191206 Comprehensions - Slightly Loopy
# ßhould return all the square roots to 100.
print([x * x for x in range(10)])

# Should return jus tthe ones that are devisible by 3.
print([x * x for x in range(10) if x % 3 == 0])

# looping and pairing
girls = ['alice', 'bernice', 'clarice']
boys = ['chris', 'arnold', 'bob']
print('Should pair the boys and girls by first letter of name:')
print([b+'+'+g for b in boys for g in girls if b[0] == g[0]])

# page 93 doing it better by doing a dictionary
girls = ['alice', 'bernice', 'clarice'] 
boys = ['chuck', 'aaron', 'bill'] 
letterGirls = {}
for girl in girls:
    letterGirls.setdefault(girl[0], []).append(girl) 
print('Should pair up aaron, bill and chuck using a dictionary:')
print([b+'+'+g for b in boys for g in letterGirls[b[0]]])

# Page 93 dictionary comprehension
squares = {i:"{} squared is {}".format(i, i**2) for i in range(10)}
print(squares[8])

# reset my global email address
# fixed my ssh keys again. 

# linking my github again