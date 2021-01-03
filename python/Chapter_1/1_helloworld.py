# Beginning Python (From Novice to Professional)
# Chapter 1
message = "Hello Python world!"
print(message)

message = "Hello Python Crash Course world!"
print(message)

mesage = "Hello Python Crash Course reader!" 
print(mesage)

# Page 17 Long Strings
print('''This is a very long string. It continues here.
And it's not over yet. "Hello, World!"
Still here.''')

# Page 18 Raw Strings
print('C:\\Program FIles\\Vendor\\Application\\execute.exe')

# Page 19 More Raw Strings
print(r'C:\Program Files\Vendor\Application\HelloWorld.exe')
# This one failed so I had to comment it out. 
# print(r'C:\Program Files\Vendor\Application\')
# Unicode, this shows up when run in python terminal but not in normal run.
"\u00C7"

# Page 20 Encoding
"Hello, world!" .encode("ASCII")
"Hello, world!" .encode("UTF-8")
"Hello, world!" .encode("UTF-32")
