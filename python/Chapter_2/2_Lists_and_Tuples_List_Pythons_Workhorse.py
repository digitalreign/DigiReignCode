# 20190818 BeginningPython Chapter 2 

# 20190818 page 34
# List's: Python's Workhorse

# The List function
months = [
'January',
'February',
'March',
'April',
'May',
'June',
'July',
'August',
'September',
'October',
'November',
'December'
]
print(list('Months'))

# Changing lists
print(months)
months[1] = 'Joseuary'
print(months)

# 20190818 page 35 Deleting Elements
del months[2]
print(months)

# Assigning to slices
capital = list('Months')
print (capital)
# Change letters to capitals
capital[2:] = list('NTHS')
print (capital)
# Add ly
capital[5:] = list('ly')
print (capital)
# 20190819 page 36 append
months.append('Californiary')
print(months)
# months.clear() and months[:] = [] would clear the list and I merged the copy command with it so that I could get all results
monthsclear = months[:]
monthsclear[2] = 'Summer'
print('The Correct Months')
print(months)
print('The Rogue Months')
print(monthsclear)
print('Clearing the Rogue Months')
monthsclear[:] =[]
print('The Cleared Months')
print(monthsclear)
print('The Correct Months')
print(months)
# 20190819 page 37 counting
# 20190821 added question an print statement
print('How many Julys do we have?')
print(months.count('July'))
# 20190820 page 37 extend
# months.extend and months + alternatemonths would work.
alternatemonths = ['Sol', 'Elephant','Holiday']
months.extend(alternatemonths)
print('Extended Months')
print(months)
# insert
months.insert(2, 'March')
print('Added March back in')
print(months)
# pop
months.pop()
print('Removing Holiday')
print(months)
# 20190820 page 40 remove
months.remove('Elephant')
print('Removing Elephant')
print(months)
months.remove('Sol',)
print('Removing Sol')
print(months)
months.remove('Californiary')
print('Removing Californiary')
print(months)

# 20190821 page 40 reverse
months.reverse()
print('Reversing the Months')
print(months)
# 20190821 page 40 sort
months.sort()
print('Sorting the Months')
print(months)
# if you want to sort but keep the original you will want to copy the original and then sort the copy.
# months_sorted = months()
# months_sorted.sort()
print(sorted('Months'))
# 20190822 page 42 Advanced sort
months.sort(key=len)
print('Sorting the Months by Length')
print(months)
# Reverse Sort
months.sort(reverse=True)
print('Reverse Sorting the Months')
print(months)
