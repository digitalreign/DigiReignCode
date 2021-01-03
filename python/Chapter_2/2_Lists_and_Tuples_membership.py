# 20190815 BeginningPython Chapter 2 

# 20190815 page 33
# Check a user name and PIN code
database = [
['albert', '1234'],
['dilbert', '4242'],
['smith', '7524'],
['jones', '9843']
]
username = input('User name: ')
pin = input('PIN code: ')
if [username, pin] in database: print('Access granted')