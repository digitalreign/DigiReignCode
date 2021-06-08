# Beginning Python (From Novice to Professional)
# Chapter 2 Lists and Tuples

# Page 25 Sequence 20190811
edward = ['Edward Gumby', 42]
john = ['John Smith', 50]
database = [edward, john]
print(database)

# Page 26 Common Sequence Operations 20190811
greeting = 'Hello'
print(greeting[0])
print(greeting[1])
print(greeting[2])
print(greeting[3])
print(greeting[4])
print(greeting[-1])

# Page 27 Calendar Indexing
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
# A list with one ending for each number from 1 to 31

# Define endings to create ordinal numbers
endings = ['st', 'nd', 'rd'] + 17 * ['th'] \
+ ['st', 'nd', 'rd'] + 7 * ['th'] \
+ ['st']
# Gather inputs for year, month, day.
year = input('Year: ')
month = input('Month (1-12): ')
day = input('Day (1-31): ')
# int = integer = display asnwer in whole number form.
month_number = int(month)
day_number = int(day)
# Remember to subtract 1 from month and day to get a correct index
month_name = months[month_number-1]
ordinal = day, endings[day_number-1]
# This is not pretty but it works for now, need to get rid of the extra ( ) and ''
print(month_name,ordinal,year)

# 20190812 Page 28 Slicing
# Create a tag for a known URL.
tag = '<a href="http://www.python.org">Python web site</a>'

# Show values for the specific characters in that URL. 
print(tag[9:30])
print(tag[32:-4])

# Setting up a new series 
numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# Show values for different slices.
print(numbers [3:6])
print(numbers [0:1])
print(numbers [3:])
print(numbers [:3])
print(numbers [-3:])
print(numbers [:-3])

# 20190813 page 29 Longer Steps
# Slicing allows for individuals to step based on a value in the final position. THis is not what it is not division, just list order and skipping values.
# Print all values
print(numbers[0:20:1])
# Happens to print the odd numbers
print(numbers[0:20:2])
# Only prints 1, 4, 7, 10, 13, 16, 29
print(numbers[0:20:3])
# Only prints 1, 5, 9, 13, 17
print(numbers[0:20:4])
# This can be accomplished by just putting the final value in if you are using the full set of numbers.
print(numbers[::4])
# 20190813 Page 30
# We can't use 0, but can count backwards.
# Count from the 10th value and down by 4 steps.
print(numbers[10::-4])
# Count from 20 as the start value and down by 4 steps until you hit the 10th value.
print(numbers[:10:-4])
# You can add two different sequences together. In this case I am creating a string bacsed on the two earlier lists.
print(numbers[10::-4]+numbers[:10:-4])
# The following line failed because you can't have a list with mixed numbers and letters.
# print(month_name+ordinal+year)
# 20190813 Page 31 
# A value can be printed mulitple times 
# 20190815 changed to month_name 
print('month_name')*5

# 20190814 Sequence Multiplications page 31
# This should take the month selected and print it in the box. 
sentence = month_name
screen_width = 80
text_width = len(sentence)
box_width = text_width + 6
left_margin = (screen_width - box_width) // 2
print()
print(' ' * left_margin + '+' + '-' * (box_width-4) + '+')
print(' ' * left_margin + '| ' + ' ' * text_width + ' |')
print(' ' * left_margin + '| ' + sentence + ' |')
print(' ' * left_margin + '| ' + ' ' * text_width + ' |')
print(' ' * left_margin + '+' + '-' * (box_width-4) + '+')
print()
