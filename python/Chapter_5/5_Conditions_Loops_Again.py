# 20191002 BeginningPython Chapter 5
# Conditional, Loops, and Some Other Statements

# Loops
# 20191018 page 85 while Loops
x = 1
while x <= 100:
    print(x)
    x += 1
# 20191021 page 85-86 while Loops
name = ''
while not name:
    name = input('Please enter your name: ')
print('Hello, {}!'.format(name))
words = [name, 'is', 'an', 'ex', 'parrot']
for word in words:
    print(word)

# 20191022 page 87 Iterating Over Dictionaries
d = {'x': 1, 'y': 2, 'z': 3}
for key in d:
    print(key, 'corresponds to', d[key])

# 20191024 page 87 Parallel Iteration
order = [1, 2, 3, 4, 5]
for i in range (len(words)):
    print (words[i], 'is the', order[i], 'word')
print(list(zip(words, order)))
for words, order in zip(words, order):
    print(words, 'is', order, 'years old')

# 20191025 page 89 Reversed and Sorted Iteration
print(sorted([4, 3, 6, 8, 3]))
print(sorted('Hello, world!'))
print(list(reversed('Hello, world!')))
print(''.join(reversed('Hello, world!')))

# 20191026 page 90 Breaking Out of Loops
from math import sqrt
for n in range(99, 0, -1):
    root = sqrt(n)
    if root == int(root):
        print(n)
        break
for n in range(2, 100, 1):
    root = sqrt(n)
    if root == int(root):
        print(n)
        break

# 20191029 page 91 While True
while True:
    letters = input('Please enter a word, or hit enter to quit: ')
    if not letters: break
    # do something with the word:
    print('The word was ', letters)

# 20191121 Getting back into the swing
print("Welcome Back")

# 20191205 Page 92 else clause
from math import sqrt
for n in range(99, 81, -1):
    root = sqrt(n)
    if root == int(root):
        print(n)
        break 
else:
    print("Didn't find it!")
# added a key

# EOF

# 20200124 New Computer Setup