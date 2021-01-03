# 20190906 BeginningPython Chapter 4
# Dictionaries: When Indices Won’t Do

# 20190906 page 62
# A simple database

# A dictionary with person names as keys. Each person is represented as
# another dictionary with the keys 'phone' and 'addr' referring to their phone
# number and address, respectively.
people = {
    'January': {
        'phone': '555-3131',
        'addr': '31 January Way'
    },
    'February': {
        'phone': '555-2828',
        'addr': '28 February Drive'
    },
    'April': {
        'phone': '555-3030',
        'addr': '30 April Street'
    }
}
# Descriptive labels for the phone number and address. These will be used
# when printing the output.
labels = {
    'phone': 'phone number',
    'addr': 'address'
}
name = input('Name: ')
# Are we looking for a phone number or an address?
request = input('Phone number (p) or address (a)? ')
# Use the correct key:
if request == 'p': key = 'phone'
if request == 'a': key = 'addr'
# Only try to print information if the name is a valid key in
# our dictionary:
if name in people: print("{}'s {} is {}.".format(name, labels[key], people[name][key]))
else: print('Name not found.')

# 20190907 String Formatting with Dictionaries page 63
zipcode = {'January': '010101', 'February': '020202', 'April': '040404'}
print("January's zipcode is {January}.".format_map(zipcode))

# clear page 64
clear = {}
clear['name'] = 'January'
clear['age'] = 42
print(clear)
returned_value = clear.clear()
print(clear)
print(returned_value)

# 20190909 copy page 64
Seasons = {'Season': 'Spring', 'Months': ['March', 'April', 'May']}
print('The Old Seasons')
print(Seasons)
New_Seasons = Seasons.copy()
New_Seasons['Season'] = 'Summer'
New_Seasons['Months'].remove('March')
print('The New Seasons')
print(New_Seasons)

# 20190910 fromkeys page 65-66
{}.fromkeys(['day', 'appointments'])
week = {}
print('Should return None')
print(week.get('day'))

# 20190912 get page 66

# A simple database using get()
# Insert database (people) from Listing 4-1 here.
labels = {
'phone': 'phone number',
'addr': 'address'
}
name = input('Name: ')
# Are we looking for a phone number or an address?
request = input('Phone number (p) or address (a)? ')
# Use the correct key:
key = request # In case the request is neither 'p' nor 'a'
if request == 'p': key = 'phone'
if request == 'a': key = 'addr'
# Use get to provide default values:
person = people.get(name, {})
label = labels.get(key, key)
result = person.get(key, 'not available')
print("{}'s {} is {}.".format(name, label, result))

# 20190913 get page 67

d = {'title': 'Python Web Site', 'url': 'http://www.python.org', 'spam': 0}
print('Print the full dictionary')
print(d.items())
it = d.items()
print('How many items are in the item dictionary?')
print(len(it))
print('Does a dictionary item have spam in it?')
print(('spam', 0) in it)

# 20190915 pop page 68
d.pop('url')
print('Remove the URL')
print(d.items())
print('Remove a random element')
d.popitem()
print(d.items())

# 20190917 setdefault page 68
d.setdefault('name', 'N/A')
print(d)
d['name'] = 'Wakka Wakka'
print('Should Print Wakka Wakka')
print(d)
d.setdefault('name', 'N/A')
print('Should Print Wakka Wakka')
print(d)

# 20190918 update page 69
x = {'title': 'Totally Not a Python Website'}
print('The value of x:')
print(x)
d.update(x)
print('The Updated version of d:')
print(d)

# 20190920 value page 69
print('Print the Values of d:')
print(d.values())
d[3] = 'https://totallynotapythonwebsite.com'
d[4] = 'September 20, 2019'
print('Print the two additional Values of d:')
print(d.values())

