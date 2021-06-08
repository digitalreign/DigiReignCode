# 20191002 BeginningPython Chapter 5
# Conditional, Loops, and Some Other Statements

# 20191002 page 76-77 Conditions and Conditional Statements

# True and False are just 1 and 0.
print('Should return 11:')
print(True + 10)
print('Should return True:')
print(bool('Something here'))

# 20191003 page 78 if and else clauses
name = input('What is your name? ')
if name.endswith('Reign'):
# 20191004 page 78-79 elif and Nesting Blocks
    if name.startswith('Mr.'):
        print('Hello Mr. Reign')
    elif name.startswith('Mrs.'):
        print('Hello Mrs. Reign')
    else:
        print('Hello, Reign')
else:
    print('Hello, Stranger')
status = "friend" if name.endswith("Reign") else "stranger"
print(status)

# 20191007 page 81 is The Identity Operator
x = y = [1, 2,3]
z = [1, 2,3]
print("Three truths and a lie:")
print(x == y)
print(x == z)
print(x is y)
# is checks for identity and not equality
print(x is z)

# 20191009 page 82 in The Membership Operator
if 'y' in name:
    print('Your name contains the letter "a".')
else:
    print('Your name does not contain the letter "y".')

# 20191012 page 83 Boolean Operators
number = int(input('Enter a number between 1 and 10: '))
if number <= 10:
    if number >= 1:
        print('Great!')
    else:
        print('Wrong!')
else:
    print('Wrong!')

# 20191013 page 83 Boolean Operators and assertions
if number <= 10 and number >= 1:
    print('Great! Great!')
else:
    print('Wrong! Wrong!')
age = int(input('Enter your age:'))
assert 0 < age < 100, 'Come on, pick a real age.'
