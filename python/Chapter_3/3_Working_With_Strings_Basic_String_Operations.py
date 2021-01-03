# 20190823 BeginningPython Chapter 3
# Working with Strings

# 20190823 page 45
# String Formatting
format = "Hello, %s. %s enough for ya?"
values = ('world', 'Hot')
print(format % values)

from string import Template
tmpl = Template("Hello, $who! $what enough for ya?")
print(tmpl.substitute(who="Mars", what="Dusty"))

# Creating a list then manipulating it.
print("{}, {}, {}, {}, {}, {} and {}".format('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
print("{1}, {3}, {5}, {0}, {2}, {4} and {6}".format('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

# 20190823 page 46
# Grabbing values and inputting them into a statement.
# Removed the pi portion

# 20190824 page 47
# Replacement Field Names
print("{first} {} {third} {}".format('Tuesday', 'Thursday', third='Wednesday', first='Monday'))
fullname = ["Monday", "Wednesday"]
print("Hello Mr. {name[1]}".format(name=fullname))
# Page 48
# Binary Conversion
print("{num} is {num:b} in binary".format(num=42))

# 20190826 page 49 Width, Precision, and Thousands Separators
print("{num:9}".format(num=9))
print("{num:8}".format(num=8))
print("{num:7}".format(num=7))
print("{num:6}".format(num=6))
print("{num:5}".format(num=5))
print("{num:4}".format(num=4))
print("{num:3}".format(num=3))
print("{num:2}".format(num=2))
print("{num:1}".format(num=1))
# num 1 and num 0 are the same
print("{num:0}".format(num=0))
print("{name:10}".format(name="January"))
print('Should print only the first 5 letters of January')
# 20190826 pg 50 Precision and separators
print("{:.5}".format("January, February, March"))
print('This is one googol.')
print('{:,}'.format(10**100))
# 20190827 pg 50 Signs, Alignment
print('{0:<10}\n{0:^10}\n{0:>10}'.format('WIN BIG'))
print("{:$^15}".format(" WIN BIG "))
# String formatting
# pg 52 Print a formatted price list with a given width
width = int(input('Please enter width that is greater than 10: '))
price_width = 10
item_width = width - price_width
header_fmt = '{{:{}}}{{:>{}}}'.format(item_width, price_width)
fmt = '{{:{}}}{{:>{}.2f}}'.format(item_width, price_width)
print('=' * width)
print(header_fmt.format('Item', 'Price'))
print('-' * width)
print(fmt.format('Apples', 0.4))
print(fmt.format('Pears', 0.5))
print(fmt.format('Cantaloupes', 1.92))
print(fmt.format('Dried Apricots (16 oz.)', 8))
print(fmt.format('Prunes (4 lbs.)', 12))
print('=' * width)
